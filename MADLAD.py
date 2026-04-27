# -*- coding: utf-8 -*-
"""
Full Fine-Tuning Script for google/madlad400-10b-mt (Arabic to French).

This script performs a full-scale fine-tuning of Google's 10B parameter
multilingual translation model for Islamic literature translation.
It is adapted from an NLLB fine-tuning script and includes critical 
adjustments for the MADLAD model architecture and its immense size.

Key Features:
- Handles a 10B parameter model efficiently using:
  - bfloat16 mixed-precision training
  - device_map="auto" for intelligent hardware utilization
  - Gradient checkpointing to save memory
  - Very small batch size (1) with high gradient accumulation (32)
- Correctly formats input for MADLAD with the required `<2lang_code>` prefix.
- Implements a custom evaluation metric using Unbabel's COMET score.
"""

# Step 0: Import All Necessary Libraries
# ==============================================================================
import torch
import numpy as np
import unbabel_comet as comet
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import os
import json

# --- Configuration ---
MODEL_CHECKPOINT = "google/madlad400-10b-mt"
COMET_MODEL_CHECKPOINT = "Unbabel/wmt22-comet-da"
TRAIN_FILE = "translations.jsonl"
VALIDATION_FILE = "test.jsonl"
# ISO 639-3 code for French
TARGET_LANG_CODE = "fra"

# Main execution block
if __name__ == "__main__":

    # Step 1: Loading Pre-Split Datasets
    # ==============================================================================
    print("\n--- Step 1: Loading Pre-Split Datasets ---")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Validation file: {VALIDATION_FILE}")
    
    # Check if files exist
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file '{TRAIN_FILE}' not found. Please run to_json.py first.")
    if not os.path.exists(VALIDATION_FILE):
        raise FileNotFoundError(f"Validation file '{VALIDATION_FILE}' not found. Please run to_json.py first.")
    
    # Load pre-split datasets
    data_files = {
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE
    }
    split_dataset = load_dataset('json', data_files=data_files)

    print(f"Datasets loaded successfully.")
    print(f"Training set size: {len(split_dataset['train'])}")
    print(f"Validation set size: {len(split_dataset['validation'])}")

    # Step 2: Loading Tokenizer and Model
    # ==============================================================================
    print("\n--- Step 2: Loading Tokenizer and Model ---")
    
    # MADLAD uses a T5-based tokenizer and does not require src_lang/tgt_lang args.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # WARNING: Loading a 10B parameter model requires significant resources.
    # The following settings are ESSENTIAL for this to run on most consumer or
    # prosumer multi-GPU setups.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Tokenizer and 10B MADLAD Model loaded successfully.")

    # Step 3: Preprocessing the Data for MADLAD
    # ==============================================================================
    print("\n--- Step 3: Preprocessing Data for MADLAD ---")
    
    # MADLAD's context length is larger; we can leverage that.
    max_length = 1024
    lang_prefix = f"<2{TARGET_LANG_CODE}>"

    def preprocess_function(examples):
        """Tokenizes source and target texts for the model."""
        # CRITICAL: Prepend the target language token to the source text.
        # This is the required format for the MADLAD model.
        prefixed_inputs = [lang_prefix + " " + text for text in examples["source_text"]]
        
        inputs = tokenizer(
            prefixed_inputs,
            max_length=max_length,
            truncation=True
        )
        
        # The target text does not need a prefix.
        # The `as_target_tokenizer` context manager is still good practice.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=max_length,
                truncation=True
            )
        
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_datasets = split_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=split_dataset["train"].column_names
    )
    
    print("Data preprocessing complete with MADLAD-specific formatting.")
    print(f"Example of a processed input: {tokenizer.decode(tokenized_datasets['train'][0]['input_ids'])}")

    # Step 4: Setting up the COMET Evaluator
    # ==============================================================================
    print("\n--- Step 4: Setting up COMET for Evaluation ---")
    comet_path = comet.download_model(COMET_MODEL_CHECKPOINT)
    comet_model = comet.load_from_checkpoint(comet_path)
    print("COMET model loaded successfully.")

    def compute_metrics(eval_preds):
        """Computes the COMET score, handling MADLAD's input format."""
        preds, labels, inputs = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Decode inputs and REMOVE the language prefix for COMET's source.
        decoded_inputs_with_prefix = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        decoded_inputs = [text.replace(lang_prefix, "").strip() for text in decoded_inputs_with_prefix]

        data_for_comet = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(decoded_inputs, decoded_preds, decoded_labels)
        ]
        
        model_output = comet_model.predict(data_for_comet, batch_size=8, gpus=torch.cuda.device_count())
        mean_score = model_output.get_scalar_mean()
        return {"comet_score": mean_score}


    # Step 5: Configuring the Trainer for a 10B Model
    # ==============================================================================
    print("\n--- Step 5: Configuring the Trainer for a 10B Model ---")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # These training arguments are adjusted for the immense size of the 10B model.
    # A small batch size and large gradient accumulation are key to fitting this in memory.
    training_args = Seq2SeqTrainingArguments(
        output_dir="madlad400-10b-finetuned-ar-to-fr-islamic",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        # CRITICAL: Batch size MUST be small. 1 is the safest.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        # CRITICAL: Increase gradient accumulation to maintain effective batch size.
        # Effective batch size = 1 (batch_size) * 32 (grad_accum) = 32
        gradient_accumulation_steps=32,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        # --- Efficiency ---
        bf16=True,
        gradient_checkpointing=True,
        # --- Metric Configuration ---
        include_inputs_for_metrics=True,
        load_best_model_at_end=True,
        metric_for_best_model="comet_score",
        greater_is_better=True,
        # --- Logging ---
        logging_dir="./logs_madlad",
        logging_strategy="steps",
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Trainer configured. Ready to start training.")

    # Step 6: Start Training
    # ==============================================================================
    print("\n--- Step 6: Starting Full Model Fine-Tuning for MADLAD-10B ---")
    print("WARNING: This will be a very long and resource-intensive process.")
    
    trainer.train()

    print("\n--- Training complete! ---")
    print("The best model checkpoint has been saved to the output directory.")