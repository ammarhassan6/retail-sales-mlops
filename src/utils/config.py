"""Load YAML config files and build absolute paths from the project root."""

from pathlib import Path

import yaml

# src/utils/config.py -> parents[2] is the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(file_name: str = "pipeline_config.yaml") -> dict:
    """Read a YAML file from the configs/ folder and return it as a dict."""
    config_path = PROJECT_ROOT / "configs" / file_name
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative_path: str) -> Path:
    """Turn a path like 'data/raw/train.csv' into an absolute path."""
    return PROJECT_ROOT / relative_path
