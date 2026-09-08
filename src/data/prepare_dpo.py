from src.data.common import parser, write_cleaned


def normalize(row):
    keys = ("prompt", "chosen", "rejected")
    value = {key: str(row.get(key, "")).strip() for key in keys}
    return value if all(value.values()) and value["chosen"] != value["rejected"] else None


if __name__ == "__main__":
    args = parser("Normalize and de-duplicate DPO JSONL").parse_args()
    print(write_cleaned(args.input, args.output, normalize, args.max_chars))
