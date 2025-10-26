from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from loguru import logger


class PlanningConfig:
    """Lightweight YAML-backed configuration for the planning layer."""

    def __init__(self, yaml_path: Optional[str | Path] = None) -> None:
        default_path = (
            Path(__file__).resolve().parent.parent.parent
            / "config"
            / "planning_config.yaml"
        )
        self._path = Path(yaml_path) if yaml_path else default_path
        with self._path.open("r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        node: Any = self._cfg
        for p in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(p, default)
        return node

    def working_crs(self) -> str:
        return self.get("crs.working", "EPSG:4326")

    def base_resolution_m(self) -> float:
        return float(self.get("resolution.base_meter", 2.0))

    @property
    def raw(self) -> Dict[str, Any]:
        return self._cfg


def setup_planning_layer(yaml_path: Optional[str | Path] = None) -> Tuple[PlanningConfig, Any]:
    """Initialize config and logger for downstream modules."""
    config = PlanningConfig(yaml_path)
    logger.info("Planning layer initialized | CRS={} | base_res={} m", config.working_crs(), config.base_resolution_m())
    return config, logger


