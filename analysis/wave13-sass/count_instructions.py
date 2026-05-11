"""Count SASS instruction frequencies and emit instruction_counts.csv."""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path("/home/codeseys/cuda-exploration/analysis/wave13-sass")

# (sass-file-stem, impl-label, kernel-label)
KERNELS = [
    ("cutile_reduction",              "cutile", "reduce_sum"),
    ("oxide_reduction",               "oxide",  "reduce_sum"),
    ("cuda_reduction",                "cuda",   "reduce_sum"),
    ("cutile_matmul_tiled",           "cutile", "matmul_tiled"),
    ("cutile_matmul_tiled_simple",    "cutile", "matmul_tiled_simple"),
    ("oxide_matmul_tiled_microtile",  "oxide",  "matmul_tiled_microtile"),
    ("cuda_matmul_tiled",             "cuda",   "matmul_tiled"),
]

# Each column: (name, regex). SASS mnemonics are whitespace-delimited and
# followed by a '.' or ' '. We match on the token after the address-col.
# cuobjdump format has the mnemonic at a fixed position but we rely on regex
# to find instances.
PATTERNS = [
    ("FFMA",            r"\bFFMA\b"),
    ("HMMA",            r"\bHMMA\b"),
    ("BMMA",            r"\bBMMA\b"),
    ("IMMA",            r"\bIMMA\b"),
    ("TCGEN05",         r"\bTCGEN05\b"),
    ("FADD",            r"\bFADD\b"),
    ("FMUL",            r"\bFMUL\b"),
    ("SHFL",            r"\bSHFL\b"),
    ("ATOM",            r"\bATOM(S|G)?\b"),
    ("BAR_SYNC",        r"\bBAR\.SYNC\b"),
    ("LDG_E",           r"\bLDG\.E\b"),
    ("LDG_E_CONSTANT",  r"\bLDG\.E\.CONSTANT\b"),
    ("LDG_128",         r"\bLDG\.E\.128\b"),
    ("LDG_64",          r"\bLDG\.E\.64\b"),
    ("LDS",             r"\bLDS\b"),
    ("STS",             r"\bSTS\b"),
    ("STG",             r"\bSTG\b"),
    ("BRA",             r"\bBRA\b"),
    ("IADD3",           r"\bIADD3\b"),
    ("HFMA2",           r"\bHFMA2\b"),
    ("UTMALDG",         r"\bUTMALDG\."),
    ("LD_E",            r"\bLD\.E\b"),
    ("LDL",             r"\bLDL\."),
    ("STL",             r"\bSTL\."),
    ("SYNCS",           r"\bSYNCS\."),
    ("LDS_128",         r"\bLDS\.128\b"),
    ("STS_128",         r"\bSTS\.128\b"),
]

def count_patterns(text: str) -> dict[str, int]:
    out = {}
    for name, rx in PATTERNS:
        out[name] = len(re.findall(rx, text))
    return out

rows = []
for stem, impl, kernel in KERNELS:
    p = ROOT / f"{stem}.sass"
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    txt = p.read_text()
    counts = count_patterns(txt)
    total_lines = txt.count("\n")
    row = {"sass_file": stem, "impl": impl, "kernel": kernel,
           **counts, "total_lines": total_lines}
    rows.append(row)
    print(f"{stem}: lines={total_lines} FFMA={counts['FFMA']} HMMA={counts['HMMA']} "
          f"BMMA={counts['BMMA']} SHFL={counts['SHFL']} ATOM={counts['ATOM']} "
          f"BAR={counts['BAR_SYNC']} LDG.E={counts['LDG_E']} LDG.128={counts['LDG_128']} "
          f"LDS={counts['LDS']} STS={counts['STS']}")

# Write CSV. Columns per task spec.
fieldnames = ["sass_file", "impl", "kernel",
              "FFMA", "HMMA", "BMMA", "IMMA", "TCGEN05",
              "FADD", "FMUL", "HFMA2", "SHFL", "ATOM", "BAR_SYNC", "SYNCS",
              "LDG_E", "LDG_E_CONSTANT", "LDG_128", "LDG_64", "LD_E",
              "UTMALDG", "LDL", "LDS", "LDS_128",
              "STS", "STS_128", "STG", "STL",
              "BRA", "IADD3",
              "total_lines"]
out_csv = ROOT / "instruction_counts.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {out_csv}")
