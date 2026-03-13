import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: str) -> ConfigDict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must parse into a dictionary.")
    return cfg


def save_config(config: ConfigDict, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def build_parser(
    description: str,
    extra_arg_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    if extra_arg_fn is not None:
        extra_arg_fn(parser)
    return parser


def parse_config_args(
    description: str,
    extra_arg_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> Tuple[ConfigDict, argparse.Namespace]:
    parser = build_parser(description, extra_arg_fn)
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg, args
