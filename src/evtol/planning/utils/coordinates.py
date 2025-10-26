from __future__ import annotations

from typing import Tuple

from pyproj import Transformer


def get_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def project_lonlat_to_xy(lon: float, lat: float, src: str, dst: str) -> Tuple[float, float]:
    transformer = get_transformer(src, dst)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


