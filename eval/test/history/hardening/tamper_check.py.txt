#!/usr/bin/env python3
"""Run the unchanged hidden-test scorer against a tampered copy of the fixed model.

The real ``data/downloads/roberta-base`` is never touched: a sibling directory
of symlinks with one edited ``config.json`` is created under
``outputs/hardening/`` and the scorer module's ``MODEL_DIR`` is pointed at it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "hardening"


def main() -> int:
    real = ROOT / "data" / "downloads" / "roberta-base"
    fake = OUT / "tampered_roberta"
    if fake.exists():
        shutil.rmtree(fake)
    fake.mkdir(parents=True)
    for item in real.iterdir():
        if item.name != "config.json":
            (fake / item.name).symlink_to(item.resolve())
    config = json.loads((real / "config.json").read_text())
    config["model_type"] = "unusable"
    (fake / "config.json").write_text(json.dumps(config, indent=2))

    spec = importlib.util.spec_from_file_location("hidden_scorer", ROOT / "eval" / "test" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_DIR = fake
    sys.argv = ["evaluate.py", "--submissions", str(OUT / "submissions_ref_only"),
                "--output", str(OUT / "scores_test_tampered.json"), "--key-sources"]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
