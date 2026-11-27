"""Build the paper tables from results/*.csv.

Usage:
    python scripts/make_tables.py [--results-dir results] [--tables-dir paper/tables] [--md results/tables.md]

Reads every CSV in the results folder (not subfolders). Each file is named
YYYY-MM-DD_<condition>_<note>.csv and has the columns in results/README.md.
Optional columns: format, split, checkpoint_step, run_id, condition (a
`condition` column overrides the name in the file name).

Writes LaTeX tables to paper/tables/*.tex and all tables as markdown to
results/tables.md. With no CSVs it writes placeholder tables. Standard
library only, so it runs in any Python 3.9+.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KNOWN_CONDITIONS = ("agent_plus_policy", "policy_only", "agent_only")  # longest first
REQUIRED = ("task", "episode", "seed", "success", "seconds", "llm_calls",
            "input_tokens", "output_tokens", "cost_usd", "code_commit")
NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)\.csv$")
NO_RESULTS = "No results yet"


# ---------------------------------------------------------------- loading

def parse_filename(name: str) -> tuple[str | None, str | None, str | None]:
    """'2026-11-28_agent_plus_policy_smoke.csv' -> ('2026-11-28', 'agent_plus_policy', 'smoke')."""
    m = NAME_RE.match(name)
    if not m:
        return None, None, None
    date, rest = m.groups()
    for c in KNOWN_CONDITIONS:
        if rest == c or rest.startswith(c + "_"):
            return date, c, rest[len(c) + 1:] or None
    head, _, note = rest.partition("_")
    return date, head, note or None


def parse_bool(v) -> bool | None:
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true", "t", "yes", "y"):
        return True
    if s in ("0", "0.0", "false", "f", "no", "n"):
        return False
    return None


def parse_float(v) -> float | None:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def load_rows(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (rows, file_info). Rows get `condition`, `date`, `file` added."""
    rows, files = [], []
    for path in sorted(Path(results_dir).glob("*.csv")):
        date, cond, note = parse_filename(path.name)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = [c.strip() for c in (reader.fieldnames or [])]
            missing = [c for c in REQUIRED if c not in cols]
            n = 0
            for raw in reader:
                r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
                ok = parse_bool(r.get("success"))
                if ok is None:
                    continue  # skip rows with no usable success value
                r["success"] = ok
                r["condition"] = r.get("condition") or cond or "unknown"
                r["date"], r["file"] = date, path.name
                rows.append(r)
                n += 1
        commits = sorted({r.get("code_commit", "") for r in rows if r["file"] == path.name} - {""})
        files.append({"file": path.name, "condition": cond, "note": note, "rows": n,
                      "missing": missing, "commits": commits})
    return rows, files


# ---------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def success_stats(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi}


def mean_of(rows: list[dict], col: str) -> float | None:
    vals = [x for x in (parse_float(r.get(col)) for r in rows) if x is not None]
    return sum(vals) / len(vals) if vals else None


def group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) or "" for k in keys)].append(r)
    return dict(out)


def fmt_rate(s: dict) -> str:
    """'40.0 [16.8, 68.7] (n=10)'"""
    if s["n"] == 0:
        return "-"
    return f"{100 * s['rate']:.1f} [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}] (n={s['n']})"


def fmt_num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def step_key(s: str):
    x = parse_float(s)
    return (0, x) if x is not None else (1, s)


# ---------------------------------------------------------------- tables
# Each table is a dict: name, caption, header, rows (list of list of str), note.

def table_per_task(rows: list[dict]) -> dict:
    conds = sorted({r["condition"] for r in rows})
    by = group(rows, "task", "condition")
    tasks = sorted({r["task"] for r in rows})
    body = [[t] + [fmt_rate(success_stats(by.get((t, c), []))) for c in conds] for t in tasks]
    if tasks:
        body.append(["All tasks"] + [fmt_rate(success_stats([r for r in rows if r["condition"] == c])) for c in conds])
    return {"name": "per_task", "caption": "Success rate (\\%) per task and condition, with Wilson 95\\% interval and episode count.",
            "header": ["Task"] + conds, "rows": body, "wide": True}


def table_format(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("format")]
    has_split = any(r.get("split") for r in rows)
    splits = sorted({r.get("split") or "all" for r in rows}) if has_split else ["all"]
    fmts = sorted({r["format"] for r in rows})
    body = []
    for f in fmts:
        cells = [f]
        for s in splits:
            sub = [r for r in rows if r["format"] == f and (not has_split or (r.get("split") or "all") == s)]
            cells.append(fmt_rate(success_stats(sub)))
        body.append(cells)
    header = ["Format"] + (splits if has_split else ["Success"])
    return {"name": "format_success",
            "caption": "Success rate (\\%) by command format, with Wilson 95\\% interval and episode count. Policy retrained per format.",
            "header": header, "rows": body}


def table_checkpoint(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("checkpoint_step")]
    by = group(rows, "format", "checkpoint_step")
    keys = sorted(by, key=lambda k: (k[0], step_key(k[1])))
    body = [[f or "-", s, fmt_rate(success_stats(by[(f, s)]))] for f, s in keys]
    return {"name": "checkpoint_success", "caption": "Success rate (\\%) by training step.",
            "header": ["Format", "Step", "Success"], "rows": body}


def table_agent_summary(rows: list[dict]) -> dict:
    order = {c: i for i, c in enumerate(("policy_only", "agent_only", "agent_plus_policy"))}
    by = group(rows, "condition")
    body = []
    for (c,) in sorted(by, key=lambda k: (order.get(k[0], 99), k[0])):
        rs = by[(c,)]
        body.append([c, fmt_rate(success_stats(rs)), fmt_num(mean_of(rs, "seconds")),
                     fmt_num(mean_of(rs, "llm_calls")), fmt_num(mean_of(rs, "input_tokens"), 0),
                     fmt_num(mean_of(rs, "output_tokens"), 0), fmt_num(mean_of(rs, "cost_usd"), 4)])
    return {"name": "agent_summary",
            "caption": "Agent alone vs.\\ agent plus policy. Success (\\%, Wilson 95\\% interval, episodes) and per-episode means.",
            "header": ["Condition", "Success", "Seconds", "LLM calls", "In tokens", "Out tokens", "Cost (USD)"],
            "rows": body, "wide": True}


def build_tables(rows: list[dict]) -> list[dict]:
    tabs = [table_agent_summary(rows), table_format(rows), table_checkpoint(rows), table_per_task(rows)]
    reasons = {"format_success": "no format column in any result",
               "checkpoint_success": "no checkpoint_step column in any result"}
    for t in tabs:
        if not t["rows"]:
            t["placeholder"] = NO_RESULTS if not rows else f"{NO_RESULTS} ({reasons.get(t['name'], 'no rows')})"
    return tabs


# ---------------------------------------------------------------- output

def tex_escape(s: str) -> str:
    s = str(s).replace("\\", "\x00")  # placeholder, so the braces below stay intact
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s.replace("\x00", r"\textbackslash{}")


def to_latex(t: dict) -> str:
    env = "table*" if t.get("wide") else "table"
    ncol = len(t["header"])
    lines = [f"% Generated by scripts/make_tables.py. Do not edit by hand.",
             f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             f"\\caption{{{t['caption']}}}", f"\\label{{tab:{t['name']}}}",
             "\\begin{adjustbox}{max width=\\linewidth}",  # shrink only, never enlarge
             f"\\begin{{tabular}}{{l{'c' * (ncol - 1)}}}", "\\toprule",
             " & ".join(tex_escape(h) for h in t["header"]) + " \\\\", "\\midrule"]
    if t.get("placeholder"):
        lines.append(f"\\multicolumn{{{ncol}}}{{c}}{{\\emph{{{tex_escape(t['placeholder'])}}}}} \\\\")
    else:
        for i, r in enumerate(t["rows"]):
            if t["name"] == "per_task" and i == len(t["rows"]) - 1:
                lines.append("\\midrule")
            lines.append(" & ".join(tex_escape(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\end{{{env}}}", ""]
    return "\n".join(l for l in lines if l != "")+"\n"


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def to_markdown(t: dict) -> str:
    cap = t["caption"].replace("\\%", "%").replace("\\ ", " ").replace("vs.\\", "vs.")
    out = [f"## {t['name']}", "", cap, "",
           "| " + " | ".join(md_escape(h) for h in t["header"]) + " |",
           "|" + "---|" * len(t["header"])]
    if t.get("placeholder"):
        out.append("| " + " | ".join([f"_{t['placeholder']}_"] + [""] * (len(t["header"]) - 1)) + " |")
    else:
        out += ["| " + " | ".join(md_escape(c) for c in r) + " |" for r in t["rows"]]
    return "\n".join(out) + "\n"


def sources_markdown(files: list[dict]) -> str:
    if not files:
        return "## Sources\n\nNo CSVs found.\n"
    out = ["## Sources", "", "| File | Condition | Rows | Commits | Missing columns |", "|---|---|---|---|---|"]
    for f in files:
        out.append(f"| {f['file']} | {f['condition'] or '-'} | {f['rows']} | "
                   f"{', '.join(f['commits']) or '-'} | {', '.join(f['missing']) or '-'} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--tables-dir", type=Path, default=REPO / "paper" / "tables")
    ap.add_argument("--md", type=Path, default=REPO / "results" / "tables.md")
    a = ap.parse_args(argv)

    rows, files = load_rows(a.results_dir)
    tabs = build_tables(rows)
    a.tables_dir.mkdir(parents=True, exist_ok=True)
    for t in tabs:
        (a.tables_dir / f"{t['name']}.tex").write_text(to_latex(t))
    md = ["# Result tables", "",
          "Generated by `scripts/make_tables.py` from `results/*.csv`. Do not edit by hand.",
          "Success is a percentage with a Wilson 95% interval and the episode count.", ""]
    md += [to_markdown(t) for t in tabs] + [sources_markdown(files)]
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text("\n".join(md))
    for f in files:
        if f["missing"]:
            print(f"warning: {f['file']} is missing columns {f['missing']}")
    print(f"{len(rows)} episodes from {len(files)} CSVs -> {a.tables_dir} and {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
