"""Train the manufacturing text IE model."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.text_ie.dataset import MissingTokenizerDependencyError
from site_safety_monitor.text_ie.model import MissingMLDependencyError
from site_safety_monitor.text_ie.train import load_training_config, train_text_ie_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train manufacturing text IE.")
    parser.add_argument("--config", required=True, help="Path to text IE training config YAML.")
    args = parser.parse_args()
    config = load_training_config(args.config)
    try:
        result = train_text_ie_model(config)
    except (MissingMLDependencyError, MissingTokenizerDependencyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
