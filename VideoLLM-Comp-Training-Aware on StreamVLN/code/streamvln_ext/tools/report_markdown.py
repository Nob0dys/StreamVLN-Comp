import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate markdown summary from experiments CSV.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    lines = []
    lines.append("# StreamVLN Experiment Report")
    lines.append("")
    lines.append(f"Total experiments: {len(rows)}")
    lines.append("")
    lines.append("| exp_id | episodes | SR | SPL | OS | NE |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['exp_id']} | {row['episodes']} | {float(row['sr']):.4f} | {float(row['spl']):.4f} | {float(row['os']):.4f} | {float(row['ne']):.4f} |"
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report generated -> {args.out_md}")


if __name__ == "__main__":
    main()
