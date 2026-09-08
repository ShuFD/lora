from src.data.common import parser, write_cleaned


def normalize(row):
    if isinstance(row.get("messages"), list) and row["messages"]:
        messages = [m for m in row["messages"] if m.get("role") in {"system", "user", "assistant"} and m.get("content")]
        return {"messages": messages} if len(messages) >= 2 and messages[-1]["role"] == "assistant" else None
    instruction, output = str(row.get("instruction", "")).strip(), str(row.get("output", "")).strip()
    if not instruction or not output: return None
    content = instruction + ("\n" + str(row.get("input", "")).strip() if row.get("input") else "")
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": output}]}


if __name__ == "__main__":
    args = parser("Normalize and de-duplicate SFT JSONL").parse_args()
    print(write_cleaned(args.input, args.output, normalize, args.max_chars))
