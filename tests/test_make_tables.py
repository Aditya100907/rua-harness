"""Tests for scripts/make_tables.py, on synthetic CSVs in a temp folder."""
import csv
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "make_tables", Path(__file__).resolve().parent.parent / "scripts" / "make_tables.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

BASE = ["task", "episode", "seed", "success", "seconds", "llm_calls",
        "input_tokens", "output_tokens", "cost_usd", "code_commit"]


def write_csv(path, rows, extra=()):
    cols = BASE + list(extra)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def ep(task, i, ok, **kw):
    r = {"task": task, "episode": i, "seed": i, "success": int(ok), "seconds": 10.0,
         "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "code_commit": "abc1234"}
    r.update(kw)
    return r


def run(tmp_path):
    tables, md = tmp_path / "tables", tmp_path / "tables.md"
    assert mt.main(["--results-dir", str(tmp_path / "results"), "--tables-dir", str(tables), "--md", str(md)]) == 0
    return tables, md.read_text()


def test_wilson_known_values():
    lo, hi = mt.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = mt.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = mt.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert mt.wilson(0, 0) == (0.0, 0.0)


def test_parse_filename():
    assert mt.parse_filename("2026-11-28_agent_plus_policy_smoke.csv") == ("2026-11-28", "agent_plus_policy", "smoke")
    assert mt.parse_filename("2026-11-28_policy_only_ckpt_10k.csv") == ("2026-11-28", "policy_only", "ckpt_10k")
    assert mt.parse_filename("2026-11-28_agent_only.csv") == ("2026-11-28", "agent_only", None)
    assert mt.parse_filename("notes.csv") == (None, None, None)


def test_zero_csvs_writes_placeholders(tmp_path):
    (tmp_path / "results").mkdir()
    tables, md = run(tmp_path)
    for name in ("per_task", "format_success", "checkpoint_success", "agent_summary"):
        tex = (tables / f"{name}.tex").read_text()
        assert "No results yet" in tex and "\\begin{tabular}" in tex
    assert md.count("No results yet") >= 4


def test_synthetic_results(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # policy_only: OpenDrawer 3/4, TurnOnMicrowave 1/4
    write_csv(res / "2026-11-28_policy_only_smoke.csv",
              [ep("OpenDrawer", i, i < 3) for i in range(4)] + [ep("TurnOnMicrowave", i, i == 0) for i in range(4)])
    # agent_only: 1/2 with tokens and cost
    write_csv(res / "2026-11-28_agent_only_smoke.csv",
              [ep("OpenDrawer", i, i == 0, seconds=60, llm_calls=4, input_tokens=1000, output_tokens=100,
                  cost_usd=0.02) for i in range(2)])
    # format sweep with split and checkpoint_step
    fmt_rows = []
    for fmt, k in (("sentence", 2), ("kv_text", 4)):
        for split, kk in (("seen", k), ("heldout", k - 1)):
            fmt_rows += [ep("OpenDrawer", i, i < kk, format=fmt, split=split, checkpoint_step=20000) for i in range(5)]
    fmt_rows += [ep("OpenDrawer", i, False, format="kv_text", split="seen", checkpoint_step=0) for i in range(5)]
    write_csv(res / "2026-11-29_policy_only_formats.csv", fmt_rows, extra=("format", "split", "checkpoint_step"))
    (res / "README.md").write_text("ignored")

    tables, md = run(tmp_path)

    # per task: OpenDrawer policy_only = 3 (smoke) + 2+1+4+3+0 (formats) = 13 of 4+25 = 29
    s = mt.success_stats([{"success": i < 13} for i in range(29)])
    assert mt.fmt_rate(s) in md
    assert "TurnOnMicrowave" in md and "25.0 [" in md  # 1/4

    # format table: kv_text seen = 4 of 5 at step 20000 plus 0 of 5 at step 0 = 4/10
    ftex = (tables / "format_success.tex").read_text()
    assert "kv\\_text" in ftex and "heldout" in ftex and "seen" in ftex
    assert "40.0 [16.8, 68.7] (n=10)" in ftex
    assert "40.0 [11.8, 76.9] (n=5)" in ftex  # sentence seen 2/5

    # checkpoint table: kv_text step 0 is 0/5, sorted before 20000
    ctex = (tables / "checkpoint_success.tex").read_text()
    assert "0.0 [0.0, 43.4] (n=5)" in ctex
    assert ctex.index("& 0 &") < ctex.index("& 20000 &")

    # agent summary: agent_only mean tokens and cost; conditions ordered
    atex = (tables / "agent_summary.tex").read_text()
    assert "agent\\_only" in atex and "1,000" in atex and "0.0200" in atex and "60.0" in atex
    assert atex.index("policy\\_only") < atex.index("agent\\_only")
    assert "No results yet" not in atex

    # sources list file names and commits
    assert "2026-11-29_policy_only_formats.csv" in md and "abc1234" in md


def test_no_format_column_gives_placeholder(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    write_csv(res / "2026-11-28_policy_only_smoke.csv", [ep("OpenDrawer", 0, True)])
    tables, _ = run(tmp_path)
    assert "no format column" in (tables / "format_success.tex").read_text()
    assert "No results yet" not in (tables / "per_task.tex").read_text()
