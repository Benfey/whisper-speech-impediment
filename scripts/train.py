import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def main():
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    # Load model and processor
    model_name = "openai/whisper-tiny"
    print(f"\nLoading model: {model_name}")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Move model to device
    model = model.to(device)
    print("Model loaded and moved to", device)

    # Load dataset
    print("\nLoading custom dataset from prepared files")
    dataset = load_from_disk("prepared_dataset")

    print(
        f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['test'])} evaluation samples")

    # Prepare the dataset
    print("\nPreparing dataset...")

    def prepare_dataset(batch):
        # Process audio
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]

        # Process text
        labels = processor(
            text=batch["text"],
            return_tensors="pt"
        ).input_ids[0]

        batch["input_features"] = input_features
        batch["labels"] = labels

        return batch

    dataset = dataset.map(
        prepare_dataset, remove_columns=dataset["train"].column_names)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="whisper-fine-tuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        warmup_steps=50,
        max_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        eval_steps=25,
        eval_strategy="steps",  # Updated from evaluation_strategy
        save_steps=50,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Data collator
    def data_collator(features):
        input_features = [{"input_features": feature["input_features"]}
            for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # Add attention mask
        batch["attention_mask"] = torch.ones(
            batch["input_features"].shape[:2],
            dtype=torch.long
        )

        labels_batch = processor.tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch

    # Metric
    import evaluate
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Update generation config
    model.generation_config.max_length = 448
    model.generation_config.suppress_tokens = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
        9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362]
    model.generation_config.begin_suppress_tokens = [220, 50257]

    # For multilingual models, explicitly set language to English
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe")

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,  # Updated from tokenizer
    )

    print("\nStarting training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model()

        prepare_dataset_with_processor,
        remove_columns = train_dataset.column_names,
        batch_size = 1
    )
    eval_dataset= eval_dataset.map(
        prepare_dataset_with_processor,
        remove_columns = eval_dataset.column_names,
        batch_size = 1
    )

    # Create data collator
    data_collator= DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training arguments optimized for small dataset
    training_args= Seq2SeqTrainingArguments(
        output_dir = "./whisper-fine-tuned",
        per_device_train_batch_size = 4,       # Reduced batch size
        per_device_eval_batch_size = 4,        # Reduced eval batch size
        gradient_accumulation_steps = 4,        # Increased gradient accumulation
        learning_rate = 5e-6,
        warmup_steps = 50,
        max_steps = 500,                       # Reduced steps for small dataset
        fp16 = True,                           # Enable mixed precision training
        gradient_checkpointing = True,         # Enable gradient checkpointing
        evaluation_strategy = "steps",
        predict_with_generate = True,
        generation_max_length = 225,
        save_steps = 25,
        eval_steps = 25,
        logging_steps = 5,
        report_to = ["tensorboard"],
        load_best_model_at_end = True,
        metric_for_best_model = "wer",
        greater_is_better = False,
        dataloader_num_workers = 0,            # Disable multiprocessing
        dataloader_pin_memory = True,          # Keep pin memory for faster data transfer
        remove_unused_columns = False,         # Keep all columns
    )

    # Create Trainer instance
    trainer= Seq2SeqTrainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = processor.feature_extractor,
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./whisper-fine-tuned/final")
    print("\nTraining completed and model saved to ./whisper-fine-tuned/final")
