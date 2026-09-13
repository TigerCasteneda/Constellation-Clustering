"""
Interactive Plotly visualisation of the Yale Bright Star Catalogue.

The plot shows the celestial sphere in 3D (unit sphere) with stars positioned
by right ascension and declination. The Big Dipper is highlighted, and a
pointer line from Merak → Dubhe → Polaris is drawn to demonstrate how to find
the North Star.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
CATALOG_PATH = DATA_ROOT / "asu.tsv"
NAMES_PATH = DATA_ROOT / "asu_names.tsv"
OUTPUT_HTML = ROOT / "celestial_sphere_plotly.html"

SPECTRAL_COLORS = {
    "O": "#9bb0ff",
    "B": "#aabfff",
    "A": "#cad7ff",
    "F": "#f8f7ff",
    "G": "#fff4ea",
    "K": "#ffd2a1",
    "M": "#ffcc6f",
}


def spectral_color(sptype: str) -> str:
    if not isinstance(sptype, str) or not sptype:
        return "#bbbbbb"
    cls = sptype.strip()[0]
    return SPECTRAL_COLORS.get(cls.upper(), "#bbbbbb")


def magnitude_to_size(mags: pd.Series) -> np.ndarray:
    filled = mags.fillna(mags.median())
    low, high = filled.quantile([0.02, 0.98])
    clipped = filled.clip(lower=low, upper=high)
    return np.interp(clipped, [clipped.min(), clipped.max()], [12, 3])


def to_cartesian(ra_deg: Iterable[float], dec_deg: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.deg2rad(np.asarray(ra_deg))
    dec = np.deg2rad(np.asarray(dec_deg))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def load_catalog() -> pd.DataFrame:
    stars = pd.read_csv(CATALOG_PATH, sep=";", skiprows=[1, 2])
    names = pd.read_csv(NAMES_PATH, sep=";", skiprows=[1, 2])
    names["Name"] = names["Name"].astype(str).str.strip()

    stars["_RAJ2000"] = pd.to_numeric(stars["_RAJ2000"], errors="coerce")
    stars["_DEJ2000"] = pd.to_numeric(stars["_DEJ2000"], errors="coerce")
    stars["Vmag"] = pd.to_numeric(stars["Vmag"], errors="coerce")
    stars = stars.dropna(subset=["_RAJ2000", "_DEJ2000"])

    stars = stars.merge(names, on="HR", how="left", suffixes=("", "_proper"))
    stars["ProperName"] = stars["Name_proper"]
    label = stars["ProperName"].fillna(stars["Name"]).replace("", np.nan)
    stars["Label"] = label.fillna("HR " + stars["HR"].astype(str))

    stars["x"], stars["y"], stars["z"] = to_cartesian(stars["_RAJ2000"], stars["_DEJ2000"])
    stars["color"] = stars["SpType"].apply(spectral_color)
    stars["size"] = magnitude_to_size(stars["Vmag"])
    return stars


def add_grid(fig: go.Figure) -> None:
    for dec in (-60, -30, 0, 30, 60):
        ra_grid = np.linspace(0, 360, 361)
        x, y, z = to_cartesian(ra_grid, np.full_like(ra_grid, dec))
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="rgba(180,180,180,0.35)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for ra in (0, 90, 180, 270):
        dec_grid = np.linspace(-90, 90, 181)
        x, y, z = to_cartesian(np.full_like(dec_grid, ra), dec_grid)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="rgba(180,180,180,0.35)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def build_figure(stars: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_grid(fig)

    fig.add_trace(
        go.Scatter3d(
            x=stars["x"],
            y=stars["y"],
            z=stars["z"],
            mode="markers",
            name="Stars",
            marker=dict(size=stars["size"], color=stars["color"], opacity=0.75),
            text=stars["Label"],
            customdata=np.stack(
                [stars["_RAJ2000"], stars["_DEJ2000"], stars["Vmag"], stars["SpType"]],
                axis=-1,
            ),
            hovertemplate="<b>%{text}</b><br>RA %{customdata[0]:.2f}°"
            "<br>Dec %{customdata[1]:.2f}°<br>Vmag %{customdata[2]:.2f}"
            "<br>SpType %{customdata[3]}<extra></extra>",
        )
    )

    dipper_order = ["Dubhe", "Merak", "Phecda", "Megrez", "Alioth", "Mizar", "Alcaid"]
    order_idx = {name: idx for idx, name in enumerate(dipper_order)}
    dipper = stars[stars["ProperName"].isin(dipper_order)].copy()
    dipper["order"] = dipper["ProperName"].map(order_idx)
    dipper = dipper.sort_values("order")

    fig.add_trace(
        go.Scatter3d(
            x=dipper["x"],
            y=dipper["y"],
            z=dipper["z"],
            mode="lines+markers+text",
            name="Big Dipper",
            marker=dict(size=10, color="#e4572e"),
            line=dict(color="#e4572e", width=4),
            text=dipper["ProperName"],
            textposition="top center",
        )
    )

    def pull_star(name: str) -> pd.Series:
        match = stars.loc[stars["ProperName"] == name]
        if match.empty:
            raise ValueError(f"Missing star '{name}' in catalog")
        return match.iloc[0]

    merak = pull_star("Merak")
    dubhe = pull_star("Dubhe")
    polaris = pull_star("Polaris")
    pointer_df = pd.DataFrame([merak, dubhe, polaris])

    fig.add_trace(
        go.Scatter3d(
            x=pointer_df["x"],
            y=pointer_df["y"],
            z=pointer_df["z"],
            mode="lines+markers+text",
            name="Pointer to Polaris",
            marker=dict(
                size=[11, 11, 16],
                color=["#ffd166", "#ffd166", "#ffd700"],
                symbol=["circle", "circle", "diamond"],
            ),
            line=dict(color="#ffd166", width=6, dash="dash"),
            text=pointer_df["ProperName"],
            textposition="top center",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[1.05],
            mode="text",
            text=["North Celestial Pole"],
            textfont=dict(color="#ffd700", size=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Celestial Sphere (Yale Bright Star Catalogue) — use Dubhe/Merak to reach Polaris",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor="white",
    )
    return fig


def main() -> None:
    stars = load_catalog()
    fig = build_figure(stars)
    fig.write_html(OUTPUT_HTML, include_plotlyjs=True)
    print(f"Saved interactive sky map to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
