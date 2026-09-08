from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from src.rl.reward import math_reward
from src.utils.logging import write_json
from src.utils.model_utils import get_quantization_config, load_tokenizer, lora_config
from src.utils.seed import set_seed


def correctness_reward(completions, answer, **_):
    def text(item):
        return item[0]["content"] if isinstance(item, list) else str(item)
    return [math_reward(text(completion), expected) for completion, expected in zip(completions, answer)]


def run(config_path: str):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=get_quantization_config(cfg["use_4bit"]), device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=cfg["train_file"], split="train")
    training = GRPOConfig(
        output_dir=cfg["output_dir"], run_name=cfg["run_name"], max_prompt_length=cfg["max_prompt_length"], max_completion_length=cfg["max_completion_length"], num_generations=cfg["num_generations"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=float(cfg["learning_rate"]), num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"], save_steps=cfg["save_steps"], bf16=True, gradient_checkpointing=True, report_to=["tensorboard"], seed=cfg["seed"],
    )
    trainer = GRPOTrainer(model=model, reward_funcs=correctness_reward, args=training, train_dataset=dataset, processing_class=tokenizer, peft_config=lora_config(cfg))
    result = trainer.train(); trainer.save_model(cfg["output_dir"]); tokenizer.save_pretrained(cfg["output_dir"])
    write_json(Path("results/grpo") / f"{cfg['run_name']}.train.json", {"config": cfg, "metrics": result.metrics})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="GRPO with exact-answer reward")
    cli.add_argument("--config", required=True); run(cli.parse_args().config)
