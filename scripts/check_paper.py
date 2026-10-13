"""Pivot Phase 8: static checks on the paper sources (no LaTeX install needed).

Checks abstract.tex + body.tex, which both wrappers (main.tex for Springer,
main_ieee.tex for IEEEtran) include, plus the wrappers themselves.

  * every \\input, \\includegraphics, \\cite, \\ref and \\n<Number> macro resolves
  * environments are balanced
  * generated tables / numbers are newer than the results they come from
  * hard-coded decimals in the prose are listed for a human to confirm
  * pre-pivot headline numbers do not appear

    python scripts/check_paper.py      (exit 1 on any failure)
"""
import re
import sys
from pathlib import Path

PAPER = Path("paper")
TEX = "\n".join((PAPER / f).read_text(encoding="utf-8")
                for f in ("main.tex", "main_ieee.tex", "abstract.tex", "body.tex"))
BODY = re.sub(r"(?<!\\)%.*", "", TEX)                       # drop comments
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


for f in re.findall(r"\\(?:tab)?input\{([^}#]+)\}", BODY):
    path = PAPER / (f if f.endswith(".tex") else f + ".tex")
    check(path.exists(), f"missing \\input file {path}")
for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", BODY):
    check(Path(f).exists(), f"missing figure {f}")

bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
for group in re.findall(r"\\cite\{([^}]+)\}", BODY):
    for k in group.split(","):
        check(k.strip() in keys, f"unknown citation {k.strip()}")

labels = set(re.findall(r"\\label\{([^}]+)\}", BODY))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", BODY))
for r in refs:
    check(r in labels, f"undefined reference {r}")
for l in labels - refs:
    check(False, f"label never referenced: {l}")

defined = set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}",
                         (PAPER / "numbers.tex").read_text() + (PAPER / "numbers_journal.tex").read_text()))
for m in set(re.findall(r"\\(n[A-Z][A-Za-z]*)", BODY)) - {"newsavebox"}:
    check(m in defined, f"undefined number macro \\{m}")

for env in set(re.findall(r"\\begin\{([^}]+)\}", BODY)):
    check(BODY.count(f"\\begin{{{env}}}") == BODY.count(f"\\end{{{env}}}"), f"unbalanced environment {env}")
check(BODY.count("{") == BODY.count("}"), "unbalanced braces")

JOURNAL_TABLES = {"reference.tex", "optimisation.tex", "sensitivity.tex"}
src = max(p.stat().st_mtime for p in Path("results/final").glob("e*.csv"))
for p in [PAPER / "numbers.tex", *PAPER.glob("tables/*.tex")]:
    if p.name not in JOURNAL_TABLES:
        check(p.stat().st_mtime >= src, f"{p} is older than results/final; rerun make_paper_assets.py")
jsrc = [q for d in ("results/journal", "results/theory") for q in Path(d).glob("*.csv")]
jsrc += [Path("results/final") / f for f in ("reference_robustness.csv", "reference_free.csv")]
jsrc = max(q.stat().st_mtime for q in jsrc if q.exists())
for p in [PAPER / "numbers_journal.tex", *(PAPER / "tables" / t for t in JOURNAL_TABLES)]:
    if p.exists():
        check(p.stat().st_mtime >= jsrc, f"{p} is older than its results; rerun make_journal_assets.py")
check("%%" not in "".join(l for l in TEX.splitlines() if l.startswith("%%") and l.rstrip().endswith("%%")),
      "unfilled %%PLACEHOLDER%% left in the paper")

# numbers from the pre-pivot draft that must not survive (F1: seed-2-only values)
for stale in ["8.1 (20.8)", "0.064", "3.7$\\times$", "p < 10^{-300}$, $R^2", "ten held-out", "upper bound"]:
    check(stale not in BODY, f"stale pre-pivot text present: {stale!r}")

prose = re.sub(r"\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}", "", BODY, flags=re.S)
prose = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}|\\begin\{align\}.*?\\end\{align\}", "", prose, flags=re.S)
hard = sorted(set(re.findall(r"(?<![\w.\\{])\d+\.\d+", prose)))
print(f"hard-coded decimals in prose (confirm by hand): {' '.join(hard)}")

if fails:
    print("FAIL\n  " + "\n  ".join(fails))
    sys.exit(1)
print(f"OK: {len(keys)} bib keys, {len(labels)} labels, {len(defined)} number macros, all resolve")
