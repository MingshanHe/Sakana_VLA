"""Summarize closed-loop best-of-N runs into one table (markdown).

    python scripts/bestofn_table.py runs/v1_stageA [tag_prefix=bon_]
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/v1_stageA")
prefix = sys.argv[2] if len(sys.argv) > 2 else "bon_"
rows = []
for d in sorted(run.glob(f"closed_loop_val_unseen_{prefix}*")):
    f = d / "results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    s, a = r["summary"], r["args"]
    rows.append((d.name.replace("closed_loop_val_unseen_", ""), a.get("candidates", 1), a.get("select", "first"),
                 a.get("tentative_stop", "hover"), s["success"], s["near_target_ever"], s["near_target_end"], s["collision_rate"],
                 s.get("stopped_rate", float("nan")), s["timeout_rate"], s["ne"], s.get("sel_changed_rate", 0.0),
                 s.get("stop_rejected", 0), s.get("stop_forced", 0), s["n"]))
print("| run | N | select | tentative stop | SR | near target ever | near target end | collision | stopped | timeout | NE (m) | changed | stop rej / forced | n |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in rows:
    print(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | **{t[4]:.2f}** | {t[5]:.2f} | {t[6]:.2f} | {t[7]:.2f} | {t[8]:.2f} | {t[9]:.2f} | "
          f"{t[10]:.2f} | {t[11]:.2f} | {t[12]} / {t[13]} | {t[14]} |")
