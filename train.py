"""
QLoRA Fine-tuning for Qwen2.5-1.5B - RUNPOD A100
WITH COMPREHENSIVE ERROR CHECKING
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
import warnings
import sys
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    output_dir: str = "/workspace/models/qwen-business-judgment"
    
    # QLoRA settings
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


def verify_file_exists(file_path: str) -> bool:
    """Verify file exists and is readable"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ ERROR: File does not exist: {file_path}")
        print(f"\n🔍 Searching for the file...")
        
        # Search in common locations
        possible_paths = [
            "/workspace/qwen-business-judgment/data/processed/training_data.jsonl",
            "/workspace/data/processed/training_data.jsonl",
            "/workspace/training_data.jsonl",
            "./data/processed/training_data.jsonl",
            "./training_data.jsonl",
        ]
        
        for p in possible_paths:
            if Path(p).exists():
                print(f"✅ Found file at: {p}")
                return p
        
        print(f"\n❌ File not found in any location!")
        print(f"\n📁 Current directory contents:")
        os.system("ls -lah /workspace/qwen-business-judgment/data/processed/ 2>/dev/null || echo 'Directory does not exist'")
        return None
    
    if not path.is_file():
        print(f"❌ ERROR: Path exists but is not a file: {file_path}")
        return None
    
    file_size = path.stat().st_size
    if file_size == 0:
        print(f"❌ ERROR: File is empty (0 bytes): {file_path}")
        return None
    
    print(f"✅ File verified: {file_path}")
    print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
    
    return file_path


class DataCollatorForCompletionOnlyLM:
    """Custom data collator"""
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        input_ids_padded = []
        labels_padded = []
        attention_mask = []
        
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        
        for ids, lbls in zip(input_ids, labels):
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                lbls = lbls[:self.max_length]
            
            padding_length = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            input_ids_padded.append(padded_ids)
            
            padded_labels = lbls + [-100] * padding_length
            labels_padded.append(padded_labels)
            
            mask = [1] * len(ids) + [0] * padding_length
            attention_mask.append(mask)
        
        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class BusinessJudgmentDataset:
    """Dataset loader with error checking"""
    
    def __init__(self, data_path: str, tokenizer, config: ModelConfig):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        
    def load_and_format(self) -> Dataset:
        print(f"📂 Loading data from {self.data_path}")
        
        # Verify file exists
        if not self.data_path.exists():
            print(f"❌ CRITICAL ERROR: Data file not found!")
            print(f"   Looking for: {self.data_path}")
            print(f"   Absolute path: {self.data_path.absolute()}")
            
            # Try to find the file
            print("\n🔍 Searching for training_data.jsonl...")
            os.system("find /workspace -name 'training_data.jsonl' -type f 2>/dev/null")
            
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Check file size
        file_size = self.data_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Data file is empty: {self.data_path}")
        
        print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Load data
        samples = []
        line_count = 0
        error_count = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                try:
                    sample = json.loads(line.strip())
                    
                    # Validate sample has required fields
                    if 'input' not in sample or 'output' not in sample:
                        print(f"⚠️  Line {line_num}: Missing 'input' or 'output' field")
                        error_count += 1
                        continue
                    
                    samples.append(sample)
                    
                    # Progress indicator
                    if line_num % 10000 == 0:
                        print(f"   Loaded {len(samples)} samples...")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: JSON decode error - {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"⚠️  Line {line_num}: Unexpected error - {e}")
                    error_count += 1
                    continue
        
        print(f"✅ Loaded {len(samples)} valid samples from {line_count} lines")
        if error_count > 0:
            print(f"⚠️  Skipped {error_count} invalid lines")
        
        if len(samples) == 0:
            raise ValueError("No valid samples loaded! Check your data file format.")
        
        # Show sample
        print(f"\n📄 Sample data point:")
        print(f"   Input: {samples[0]['input'][:100]}...")
        print(f"   Output: {samples[0]['output'][:100]}...")
        
        # Format with instruction template
        print("\n🔄 Formatting samples...")
        formatted_samples = []
        for i, sample in enumerate(samples):
            try:
                formatted_text = self._format_sample(sample)
                formatted_samples.append({'text': formatted_text})
            except Exception as e:
                print(f"⚠️  Error formatting sample {i}: {e}")
                continue
        
        print(f"✅ Formatted {len(formatted_samples)} samples")
        
        dataset = Dataset.from_list(formatted_samples)
        
        # Tokenize
        print("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            num_proc=1,  # Single process for stability
        )
        
        # Filter long sequences
        print("✂️  Filtering sequences...")
        original_size = len(tokenized_dataset)
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) <= self.config.max_seq_length
        )
        filtered_count = original_size - len(tokenized_dataset)
        
        if filtered_count > 0:
            print(f"   Filtered out {filtered_count} sequences (too long)")
        
        print(f"✅ Final dataset size: {len(tokenized_dataset)} samples")
        
        if len(tokenized_dataset) == 0:
            raise ValueError("No samples remaining after tokenization! Check max_seq_length setting.")
        
        return tokenized_dataset
    
    def _format_sample(self, sample: dict) -> str:
        template = f"""<|im_start|>system
You are an expert business analyst specializing in failure analysis and strategic judgment. Your role is to provide clear, actionable insights about business situations without using risk scores or predictions.<|im_end|>
<|im_start|>user
{sample['input']}

TASK:
Analyze the situation and provide:
1. Core failure dynamics
2. What actions could realistically work
3. What actions will not work and why
4. Final judgment<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""
        return template
    
    def _tokenize_function(self, example):
        result = self.tokenizer(
            example['text'],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result


class QLoRATrainer:
    """Training pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def load_model_and_tokenizer(self):
        print(f"📥 Loading: {self.config.model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        model, tokenizer = self.load_model_and_tokenizer()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            group_by_length=False,
            report_to="none",  # Disabled to avoid TensorBoard error
            run_name="qwen-business-judgment",
            logging_dir=None,  # Disabled
        )
        
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            max_length=self.config.max_seq_length
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("\n🚀 Starting training...")
        print(f"📊 Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"📊 Eval samples: {len(eval_dataset)}")
        
        total_steps = len(train_dataset) * self.config.num_train_epochs // (
            self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        )
        print(f"⏱️  Total steps: {total_steps}")
        print(f"⏱️  Est. time on A100: {total_steps * 0.5 / 60:.1f} minutes")
        print(f"💰 Est. cost: ${total_steps * 0.5 / 3600 * 1.89:.2f}")
        
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted! Saving...")
            trainer.save_model(self.config.output_dir + "_interrupted")
            raise
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            trainer.save_model(self.config.output_dir + "_error")
            raise
        
        print("\n💾 Saving final model...")
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        with open(f"{self.config.output_dir}/training_config.json", 'w') as f:
            json.dump({
                'model_name': self.config.model_name,
                'lora_r': self.config.lora_r,
                'epochs': self.config.num_train_epochs,
                'samples': len(train_dataset),
            }, f, indent=2)
        
        print(f"✅ Model saved to: {self.config.output_dir}")
        return model, tokenizer


def main():
    print("=" * 70)
    print("🚀 QWEN2.5-1.5B TRAINING ON RUNPOD A100")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        sys.exit(1)
    
    torch.cuda.empty_cache()
    
    # Verify data file first
    data_path = "/workspace/qwen-business-judgment/data/processed/training_data.jsonl"
    verified_path = verify_file_exists(data_path)
    
    if verified_path is None:
        print("\n❌ CRITICAL: Cannot find data file!")
        print("\n📋 Please upload training_data.jsonl to:")
        print("   /workspace/qwen-business-judgment/data/processed/")
        print("\n💡 Use Jupyter Lab Upload feature")
        sys.exit(1)
    
    config = ModelConfig()
    trainer = QLoRATrainer(config)
    model, tokenizer = trainer.load_model_and_tokenizer()
    
    print("\n📂 Loading dataset...")
    dataset_loader = BusinessJudgmentDataset(
        data_path=verified_path,
        tokenizer=tokenizer,
        config=config
    )
    
    try:
        dataset = dataset_loader.load_and_format()
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty after loading!")
        sys.exit(1)
    
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"\n📊 Training: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    model, tokenizer = trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Model: {config.output_dir}")


if __name__ == "__main__":
    main()