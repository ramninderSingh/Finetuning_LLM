"""
Inference module for QLoRA fine-tuned LLM
Loads the LoRA adapter and runs generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_finetuned_model(base_model: str, adapter_path: str):
    """Load base model + LoRA adapter for inference."""
    logger.info(f"Loading base model: {base_model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Merging LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 256):
    """Run inference with the instruction-following format."""
    if input_text:
        prompt = f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"""
    else:
        prompt = f"""### Instruction:\n{instruction}\n\n### Response:\n"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    return response


if __name__ == "__main__":
    BASE_MODEL = "microsoft/phi-2"
    ADAPTER_PATH = "./outputs/qlora-phi2"

    model, tokenizer = load_finetuned_model(BASE_MODEL, ADAPTER_PATH)

    # Test examples
    test_cases = [
        {"instruction": "Explain what a transformer model is in simple terms."},
        {"instruction": "Write a Python function to reverse a string."},
        {
            "instruction": "Summarize the following passage.",
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Instruction: {case['instruction']}")
        if "input" in case:
            print(f"Input: {case['input']}")
        response = generate(model, tokenizer, case["instruction"], case.get("input", ""))
        print(f"Response: {response}")
