"""扫描 results/logs/ 下所有实验，列成一张可检索的目录表。"""

import json
import sys
from pathlib import Path

LOG_ROOT = Path(__file__).parents[1] / "results" / "logs"


def last_line(p: Path) -> str:
    if not p.exists():
        return ""
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main(root: Path = LOG_ROOT) -> None:
    print(f"{'exp':<34} {'tag':<16} {'commit':<9} 末行")
    for d in sorted(Path(root).glob("exp_*")):
        meta = {}
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        tail = last_line(d / "run.log")[:70]
        print(
            f"{d.name:<34} {meta.get('tag', '') or '-':<16} {meta.get('git_commit', '') or '-':<9} {tail}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_ROOT)
