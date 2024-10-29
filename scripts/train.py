import os
import torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch):
    """Prepare dataset by loading audio and computing input features."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    with processor.as_target_processor():
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    """Compute WER metrics."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = evaluate.load("wer")
    return {"wer": wer.compute(predictions=pred_str, references=label_str)}


if __name__ == "__main__":
    # Load pre-trained model and processor
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    # Prepare custom dataset
    data_dir = Path("data")
    dataset = load_dataset(
        "custom_data/data_prep.py",
        data_dir=str(data_dir),
        cache_dir="cache"
    )

    # Process dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=4
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="whisper-fine-tuned",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2.5e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Train
    trainer.train()

    # Save fine-tuned model
    trainer.save_model("whisper-fine-tuned-final")
