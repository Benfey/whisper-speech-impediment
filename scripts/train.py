import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Audio, Dataset
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import json
from pathlib import Path
from functools import partial

# Configure GPU settings for optimal performance
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check CUDA availability and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Clear GPU cache
    torch.cuda.empty_cache()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """Prepare a single batch of the dataset"""
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def load_custom_dataset():
    """Load dataset from prepared audio_paths and text files"""
    audio_paths = {}
    transcripts = {}

    # Read audio_paths file
    with open('data/audio_paths', 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, path = line.strip().split(' ', 1)
            audio_paths[utt_id] = path

    # Read text file
    with open('data/text', 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, text = line.strip().split(' ', 1)
            transcripts[utt_id] = text

    # Create dataset
    dataset_dict = {
        "audio": [audio_paths[utt_id] for utt_id in audio_paths.keys()],
        "sentence": [transcripts[utt_id] for utt_id in audio_paths.keys()]
    }

    return Dataset.from_dict(dataset_dict)


if __name__ == "__main__":
    # Load the processor and model
    model_name = "openai/whisper-tiny"
    print(f"\nLoading model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Configure model for mixed precision training
    model.config.use_cache = False  # This reduces memory usage

    # Move model to GPU if available
    model = model.to(device)
    print(f"Model loaded and moved to {device}")

    # Load metric
    metric = evaluate.load("wer")

    # Load custom dataset
    print("\nLoading custom dataset from prepared files")
    dataset = load_custom_dataset()
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split dataset into train/eval
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(
        f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")

    # Create partial function with processor
    prepare_dataset_with_processor = partial(
        prepare_dataset, processor=processor)

    # Prepare dataset without multiprocessing
    print("\nPreparing dataset...")
    train_dataset = train_dataset.map(
        prepare_dataset_with_processor,
        remove_columns=train_dataset.column_names,
        batch_size=1
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset_with_processor,
        remove_columns=eval_dataset.column_names,
        batch_size=1
    )

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments optimized for small dataset
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-fine-tuned",
        per_device_train_batch_size=4,       # Reduced batch size
        per_device_eval_batch_size=4,        # Reduced eval batch size
        gradient_accumulation_steps=4,        # Increased gradient accumulation
        learning_rate=5e-6,
        warmup_steps=50,
        max_steps=500,                       # Reduced steps for small dataset
        fp16=True,                           # Enable mixed precision training
        gradient_checkpointing=True,         # Enable gradient checkpointing
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25,
        eval_steps=25,
        logging_steps=5,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,            # Disable multiprocessing
        dataloader_pin_memory=True,          # Keep pin memory for faster data transfer
        remove_unused_columns=False,         # Keep all columns
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./whisper-fine-tuned/final")
    print("\nTraining completed and model saved to ./whisper-fine-tuned/final")
