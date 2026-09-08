from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from statistics import median

from src.evaluation import code_eval, gsm8k, preference_eval
from src.utils.logging import write_json
from src.utils.model_utils import load_tokenizer


def load_rows(path: str):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate(model, tokenizer, prompts, max_new_tokens):
    import torch
    outputs, lengths, latencies = [], [], []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(rendered, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latencies.append(time.perf_counter() - start)
        new_tokens = generated[0][batch.input_ids.shape[1]:]
        lengths.append(len(new_tokens)); outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs, lengths, latencies


def main():
    cli = argparse.ArgumentParser(description="Evaluate Base/SFT/DPO/GRPO models on local JSONL task sets")
    cli.add_argument("--model", required=True); cli.add_argument("--tasks", nargs="+", default=["gsm8k"])
    cli.add_argument("--gsm8k-file"); cli.add_argument("--code-file"); cli.add_argument("--preference-file")
    cli.add_argument("--output", default=None); cli.add_argument("--max-new-tokens", type=int, default=512); cli.add_argument("--execute-code", action="store_true")
    args = cli.parse_args()
    from transformers import AutoModelForCausalLM
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
    model.eval(); metrics, examples, all_lengths, all_latencies = {}, {}, [], []
    sources = {"gsm8k": args.gsm8k_file, "code": args.code_file, "preference": args.preference_file}
    for task in args.tasks:
        source = sources.get(task)
        if not source: raise ValueError(f"--{task}-file is required for task '{task}'")
        rows = load_rows(source); predictions, lengths, latencies = generate(model, tokenizer, [r["prompt"] for r in rows], args.max_new_tokens)
        all_lengths += lengths; all_latencies += latencies; examples[task] = [{"id": r.get("id", i), "prediction": p} for i, (r, p) in enumerate(zip(rows, predictions))]
        if task == "gsm8k": metrics[task] = gsm8k.score(predictions, [r["answer"] for r in rows])
        elif task == "code": metrics[task] = code_eval.score(predictions, [r["tests"] for r in rows], args.execute_code)
        elif task == "preference": metrics[task] = preference_eval.score(predictions, [r["preferred"] for r in rows])
        else: raise ValueError(f"Unknown task: {task}")
    metrics["efficiency"] = {"samples": len(all_lengths), "avg_generated_tokens": sum(all_lengths) / len(all_lengths) if all_lengths else 0, "median_generated_tokens": median(all_lengths) if all_lengths else 0, "avg_latency_seconds": sum(all_latencies) / len(all_latencies) if all_latencies else 0}
    destination = args.output or str(Path("results") / f"{Path(args.model).name}.eval.json")
    write_json(destination, {"model": args.model, "tasks": args.tasks, "metrics": metrics, "examples": examples})
    print(json.dumps(metrics, ensure_ascii=False, indent=2)); print(f"Saved {destination}")


if __name__ == "__main__": main()
