from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from src.utils.logging import write_json
from src.utils.model_utils import get_quantization_config, load_tokenizer, lora_config
from src.utils.seed import set_seed


def run(config_path: str):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg["model_name"])
    kwargs = dict(quantization_config=get_quantization_config(cfg["use_4bit"]), device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kwargs); model.config.use_cache = False
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["reference_model_name"], **kwargs) if cfg.get("reference_model_name") else None
    dataset = load_dataset("json", data_files=cfg["train_file"], split="train")
    training = DPOConfig(
        output_dir=cfg["output_dir"], run_name=cfg["run_name"], beta=float(cfg["beta"]), max_length=cfg["max_length"], max_prompt_length=cfg["max_prompt_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_train_epochs"], logging_steps=cfg["logging_steps"], save_steps=cfg["save_steps"], save_total_limit=2,
        bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit", report_to=["tensorboard"], seed=cfg["seed"],
    )
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=training, train_dataset=dataset, processing_class=tokenizer, peft_config=lora_config(cfg))
    result = trainer.train(); trainer.save_model(cfg["output_dir"]); tokenizer.save_pretrained(cfg["output_dir"])
    write_json(Path("results/dpo") / f"{cfg['run_name']}.train.json", {"config": cfg, "metrics": result.metrics})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="QLoRA DPO")
    cli.add_argument("--config", required=True); run(cli.parse_args().config)
