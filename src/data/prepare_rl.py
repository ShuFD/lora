from src.data.common import parser, write_cleaned


def normalize(row):
    prompt, answer = str(row.get("prompt", row.get("question", ""))).strip(), str(row.get("answer", "")).strip()
    return {"prompt": prompt, "answer": answer, "id": str(row.get("id", ""))} if prompt and answer else None


if __name__ == "__main__":
    args = parser("Normalize verifiable RL prompts").parse_args()
    print(write_cleaned(args.input, args.output, normalize, args.max_chars))
