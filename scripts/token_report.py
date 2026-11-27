"""Count language tokens per command format, for the 18 tasks. Usage: python scripts/token_report.py <tokenizer.json>"""
import json, sys
from tokenizers import Tokenizer
from rua_harness import ATOMIC_SEEN, Command, to_policy_text, to_sentence

tok = Tokenizer.from_file(sys.argv[1])
n = lambda s: len(tok.encode(s).ids)   # includes special tokens, like DEM's count
rows, LIMIT = [], 32
for task, d in ATOMIC_SEEN.items():
    c = Command.from_json(d)
    rows.append((task, n(to_sentence(c)), n(json.dumps(d, separators=(",", ":"))), n(json.dumps(d)), n(to_policy_text(c)), to_policy_text(c)))
print(f"{'task':28s} sent  json-min  json  kv   kv text")
for r in rows:
    print(f"{r[0]:28s} {r[1]:4d} {r[2]:8d} {r[3]:5d} {r[4]:3d}   {r[5]}")
for i, name in enumerate(["sentence", "json-min", "json", "kv"], start=1):
    v = [r[i] for r in rows]
    print(f"{name:9s} mean {sum(v)/len(v):5.1f}  max {max(v):3d}  over {LIMIT}: {sum(x > LIMIT for x in v)}/18")
long = Command.from_json({"skill": "pick_place", "object": "small green bell pepper next to the cutting board",
                          "source": "counter", "target": "top shelf of the left cabinet"})
print("long-name stress case, kv:", n(to_policy_text(long)), "| json:", n(json.dumps({"skill": "pick_place", "object": long.object.name, "source": "counter", "target": long.target.name})))
