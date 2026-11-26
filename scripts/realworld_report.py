"""Markdown tables from scripts/realworld_eval.py results, one column per run.

    python scripts/realworld_report.py training/runs/eval/realworld/karna-base-cuda-baseline.json \\
        training/runs/eval/realworld/karna-base-cuda-beam-q10.json --names greedy "beam + LM"

Offline: CER for Khmer sets, WER for English sets, in %. Live: the same for the finals, plus missed
speech (speech in no segment, or in one whose final is empty) and decode seconds per second of audio. The last column is the relative change from the first.
"""

from __future__ import annotations

import argparse
import json


def _err(r: dict) -> float:
    return r["wer"] if "wer" in r else r["cer"]


def table(runs: list[dict], names: list[str], kind: str) -> str:
    sets = [s for s in runs[0][kind] if all(s in r[kind] for r in runs)]
    if not sets:
        return ""
    head = "| Set | Metric | " + " | ".join(names) + (" | Change |" if len(runs) > 1 else " |")
    rule = "| :--- | :--- | " + " | ".join("---:" for _ in names) + (" | ---: |" if len(runs) > 1 else " |")
    lines = [head, rule]
    for s in sets:
        vals = [_err(r[kind][s]) for r in runs]
        metric = "WER" if "wer" in runs[0][kind][s] else "CER"
        row = f"| {s} | {metric} | " + " | ".join(f"{100 * v:.1f}" for v in vals)
        if len(runs) > 1:
            row += f" | {100 * (vals[-1] / vals[0] - 1):+.0f} % |" if vals[0] else " | |"
        else:
            row += " |"
        lines.append(row)
    if kind == "live":
        for key, label, scale, fmt in (("missed_speech", "speech with no text %", 100, "{:.1f}"),
                                       ("decode_s_per_audio_s", "decode s per s", 1, "{:.3f}")):
            lines.append(f"| all streams | {label} | " + " | ".join(
                fmt.format(scale * sum(r[kind][s][key] for s in sets) / len(sets)) for r in runs)
                + (" | |" if len(runs) > 1 else " |"))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    runs = [json.load(open(p, encoding="utf-8")) for p in args.results]
    names = args.names or [r.get("tag") or p.rsplit("/", 1)[-1] for r, p in zip(runs, args.results)]
    for kind in ("offline", "live"):
        out = table(runs, names, kind)
        if out:
            print(f"**{kind.capitalize()}** ({runs[0]['model']}, {runs[0]['device']})\n\n{out}\n")


if __name__ == "__main__":
    main()
