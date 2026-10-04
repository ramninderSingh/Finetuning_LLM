"""
LLM Fine-tuning with QLoRA (LoRA + 4-bit Quantization)
Uses PEFT + BitsAndBytes + HuggingFace Transformers
Model: microsoft/phi-2 or meta-llama/Llama-3.2-1B
"""

import os
import torch
import logging
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Configuration ----------

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "microsoft/phi-2"       # Change to any HF model
    output_dir: str = "./outputs/qlora-phi2"

    # QLoRA (4-bit quantization)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"   # bfloat16 for Ampere+ GPUs
    bnb_4bit_quant_type: str = "nf4"          # NF4 = best for QLoRA
    use_double_quant: bool = True              # Nested quantization

    # LoRA Adapter
    lora_r: int = 8                           # Low-rank dimension
    lora_alpha: int = 32                      # Scaling factor (alpha/r = 4)
    lora_dropout: float = 0.05
    target_modules: list = None               # Auto-detected per model

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4      # Effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_seq_length: int = 512
    save_steps: int = 100
    logging_steps: int = 10

    # Dataset
    dataset_name: str = "tatsu-lab/alpaca"    # Instruction-following dataset
    max_samples: int = 5000                   # Use subset for fast training

    # W&B Logging
    use_wandb: bool = False
    wandb_project: str = "qlora-finetuning"

    def __post_init__(self):
        if self.target_modules is None:
            # Auto target — phi-2 uses these layers
            self.target_modules = ["q_proj", "k_proj", "v_proj", "dense"]


config = TrainingConfig()


# ---------- Setup Functions ----------

def get_bnb_config(cfg: TrainingConfig) -> BitsAndBytesConfig:
    """4-bit quantization config using NF4 + double quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )


def get_lora_config(cfg: TrainingConfig) -> LoraConfig:
    """LoRA adapter configuration."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(cfg: TrainingConfig):
    """Load quantized model + tokenizer."""
    logger.info(f"Loading model: {cfg.model_name} with 4-bit QLoRA...")

    bnb_config = get_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------- Dataset Preparation ----------

def format_alpaca_sample(sample: dict) -> str:
    """Format into instruction-following prompt template."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return prompt


def load_and_prepare_dataset(cfg: TrainingConfig):
    """Load dataset and apply formatting."""
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split="train")

    if cfg.max_samples and len(dataset) > cfg.max_samples:
        dataset = dataset.select(range(cfg.max_samples))
        logger.info(f"Using {cfg.max_samples} samples for training.")

    dataset = dataset.map(
        lambda x: {"text": format_alpaca_sample(x)},
        remove_columns=dataset.column_names,
    )

    logger.info(f"Sample formatted prompt:\n{dataset[0]['text'][:300]}...")
    return dataset


# ---------- Training ----------

def train():
    """Main training function."""
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=f"qlora-{config.model_name.split('/')[-1]}")

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_and_prepare_dataset(config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        fp16=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to="wandb" if config.use_wandb else "none",
        optim="paged_adamw_8bit",           # Memory-efficient optimizer
        group_by_length=True,               # Faster training with similar lengths
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=True,                       # Pack multiple samples into one sequence
    )

    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("Training complete!")
    return config.output_dir


if __name__ == "__main__":
    output_path = train()
    logger.info(f"Model saved at: {output_path}")
