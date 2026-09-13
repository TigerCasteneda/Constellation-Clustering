from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.io as pio
import json
from string import Template

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
CATALOG_PATH = DATA_ROOT / "asu.tsv"
NAMES_PATH = DATA_ROOT / "asu_names.tsv"
OUTPUT_HTML = ROOT / "optimized_constellations.html"
OUTPUT_HTML_INTERACTIVE = ROOT / "optimized_constellations_interactive.html"
OUTPUT_DIR = ROOT / "constellation_images"

# Filter settings
VMAG_ANCHOR_LIMIT = 4.5  # Stars brighter than this define the constellation centers
VMAG_VISIBLE_LIMIT = 6.5 # Stars brighter than this are plotted/assigned

# Visualization colors (High contrast)
import plotly.express as px
COLOR_SEQ = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet

def to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert RA/Dec to unit vectors on the sphere."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))

def great_circle_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate great circle distance (in radians) between unit vectors."""
    # Clip dot product to [-1, 1] to handle numerical errors
    dot = np.clip(np.einsum('ij,ij->i', u, v), -1.0, 1.0)
    return np.arccos(dot)

def load_data() -> pd.DataFrame:
    """Load and preprocess star catalog."""
    try:
        stars = pd.read_csv(CATALOG_PATH, sep=";", skiprows=[1, 2])
        names = pd.read_csv(NAMES_PATH, sep=";", skiprows=[1, 2])
    except FileNotFoundError:
        print(f"Error: Data files not found in {DATA_ROOT}")
        exit(1)

    names["Name"] = names["Name"].astype(str).str.strip()
    
    # Clean numeric columns
    for col in ["_RAJ2000", "_DEJ2000", "Vmag"]:
        stars[col] = pd.to_numeric(stars[col], errors="coerce")
    
    stars = stars.dropna(subset=["_RAJ2000", "_DEJ2000", "Vmag"])
    
    # Merge names
    stars = stars.merge(names, on="HR", how="left", suffixes=("", "_proper"))
    stars["ProperName"] = stars["Name_proper"]
    label = stars["ProperName"].fillna(stars["Name"]).replace("", np.nan)
    stars["Label"] = label.fillna("HR " + stars["HR"].astype(str))
    
    # Calculate vectors
    vecs = to_cartesian(stars["_RAJ2000"].values, stars["_DEJ2000"].values)
    stars["vx"] = vecs[:, 0]
    stars["vy"] = vecs[:, 1]
    stars["vz"] = vecs[:, 2]
    
    return stars.reset_index(drop=True)

class SphericalKMeans:
    """
    K-Means customized for the unit sphere using Cosine Similarity.
    """
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-6, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X: np.ndarray):
        n_samples, _ = X.shape
        
        # K-Means++ Initialization on Sphere
        # 1. Choose first centroid randomly
        self.centroids = np.zeros((self.n_clusters, 3))
        self.centroids[0] = X[self.rng.integers(n_samples)]
        
        # 2. Choose remaining centroids based on distance probability
        for i in range(1, self.n_clusters):
            # Distance is 1 - cosine_similarity (chord-like behavior for probability)
            # We use dot product. Closer to 1 means closer.
            dists = 1.0 - np.clip(X @ self.centroids[:i].T, -1, 1).max(axis=1)
            probs = dists / dists.sum()
            # Cumulative probability for selection
            cum_probs = np.cumsum(probs)
            r = self.rng.random()
            idx = np.searchsorted(cum_probs, r)
            self.centroids[i] = X[min(idx, n_samples - 1)]

        # Iteration
        for _ in range(self.max_iter):
            # E-step: Assign points to nearest centroid (max dot product)
            sims = X @ self.centroids.T
            self.labels = np.argmax(sims, axis=1)
            
            # M-step: Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = self.labels == k
                if not np.any(mask):
                    # Re-init empty cluster randomly
                    new_centroids[k] = X[self.rng.integers(n_samples)]
                else:
                    # Mean vector
                    vec_sum = X[mask].sum(axis=0)
                    # Normalize to project back to sphere
                    new_centroids[k] = vec_sum / np.linalg.norm(vec_sum)
            
            # Check convergence
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sims = X @ self.centroids.T
        return np.argmax(sims, axis=1)

def calculate_mst_lines(df_cluster: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """
    Compute Minimum Spanning Tree (MST) for a cluster to draw constellation lines.
    Returns a list of coordinates for Plotly lines.
    """
    points = df_cluster[["vx", "vy", "vz"]].values
    if len(points) < 2:
        return []
    
    # Calculate pairwise Euclidean distances (chord length)
    # MST on chord length is topologically similar to Great Circle for local clusters
    dists = squareform(pdist(points))
    mst = minimum_spanning_tree(dists).toarray()
    
    lines_x, lines_y, lines_z = [], [], []
    
    # Extract edges
    rows, cols = np.where(mst > 0)
    for r, c in zip(rows, cols):
        # Add line segment separated by None
        lines_x.extend([points[r, 0], points[c, 0], None])
        lines_y.extend([points[r, 1], points[c, 1], None])
        lines_z.extend([points[r, 2], points[c, 2], None])
        
    return lines_x, lines_y, lines_z


def calculate_mst_edges(df_cluster: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return MST edges as index pairs for 2D plotting (RA/Dec)."""
    points = df_cluster[["vx", "vy", "vz"]].values
    if len(points) < 2:
        return []
    dists = squareform(pdist(points))
    mst = minimum_spanning_tree(dists).toarray()
    rows, cols = np.where(mst > 0)
    return list(zip(rows.tolist(), cols.tolist()))


def wrap_ra_local(ra_deg: np.ndarray) -> np.ndarray:
    """Center RA around the mean direction to avoid 0/360 splits."""
    ra_rad = np.deg2rad(ra_deg)
    mean_angle = np.arctan2(np.sin(ra_rad).mean(), np.cos(ra_rad).mean())
    mean_deg = np.rad2deg(mean_angle) % 360.0
    return ((ra_deg - mean_deg + 180.0) % 360.0) - 180.0


def plot_constellation_images(visible_stars: pd.DataFrame, labels: np.ndarray, k: int) -> None:
    """Plot each constellation separately (RA/Dec) with MST lines and save PNGs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cid in range(k):
        cluster = visible_stars[visible_stars["cluster"] == cid]
        if cluster.empty:
            continue
        color = COLOR_SEQ[cid % len(COLOR_SEQ)]

        ra = cluster["_RAJ2000"].to_numpy()
        dec = cluster["_DEJ2000"].to_numpy()
        ra_wrapped = wrap_ra_local(ra)

        # MST edges on brighter stars for shape clarity
        bright = cluster[cluster["Vmag"] <= VMAG_ANCHOR_LIMIT].copy()
        bright_ra_wrapped = wrap_ra_local(bright["_RAJ2000"].to_numpy()) if not bright.empty else np.array([])
        bright_dec = bright["_DEJ2000"].to_numpy()
        edges = calculate_mst_edges(bright) if len(bright) > 1 else []

        plt.figure(figsize=(6, 5))
        sizes = np.clip((7 - cluster["Vmag"]) * 10, 6, 80)
        plt.scatter(ra_wrapped, dec, s=sizes, color=color, alpha=0.75, edgecolors="none", label=f"Constellation {cid}")

        for i, j in edges:
            plt.plot(
                [bright_ra_wrapped[i], bright_ra_wrapped[j]],
                [bright_dec[i], bright_dec[j]],
                color="white",
                linewidth=1.6,
                alpha=0.9,
            )

        # # Annotate top 3 brightest stars in cluster
        # top = cluster.nsmallest(3, "Vmag")
        # top_ra = wrap_ra_local(top["_RAJ2000"].to_numpy())
        # for (_, row), x, y in zip(top.iterrows(), top_ra, top["_DEJ2000"].to_numpy()):
        #     plt.text(x, y, row["Label"], fontsize=8, color="white", ha="center", va="bottom")

        plt.gca().invert_xaxis()  # Match sky charts (RA increases to the left)
        plt.title(f"Constellation {cid} (k={k})")
        plt.xlabel("Right Ascension (deg, wrapped)")
        plt.ylabel("Declination (deg)")
        plt.grid(True, linestyle="--", alpha=0.3, color="gray")
        plt.gca().set_facecolor("black")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"constellation_{cid:02d}.png"
        plt.savefig(out_path, dpi=200, facecolor="black")
        plt.close()
        print(f"Saved {out_path}")


def save_interactive_highlight_html(fig: go.Figure, cluster_traces: List[List[int]]) -> None:
    """
    Save an HTML with an interactive checklist to highlight selected constellations.
    Selected clusters keep original colors; others fade to gray. Clearing selection shows all.
    """
    plot_div = pio.to_html(fig, include_plotlyjs=False, full_html=False)
    checklist_items = "\n".join(
        f'<label style="margin-right:10px;"><input type="checkbox" value="{cid}"> C{cid:02d}</label>'
        for cid in range(len(cluster_traces))
    )
    cluster_traces_json = json.dumps(cluster_traces)

    template = Template("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { background: #0b0b0b; color: #eee; font-family: Arial, sans-serif; }
    #controls { padding: 10px; margin-bottom: 8px; border: 1px solid #333; border-radius: 6px; }
    label { user-select: none; }
  </style>
</head>
<body>
  <div id="controls">
    <strong>Highlight constellations:</strong> (checkbox multi-select; clear to show all)<br>
    $checklist_items
  </div>
  $plot_div
  <script>
    const clusterTraces = $cluster_traces_json;
    const checkboxes = Array.from(document.querySelectorAll('#controls input[type=checkbox]'));
    const gd = document.querySelector('.plotly-graph-div');

    function stashStyles() {
      gd.data.forEach(tr => {
        if (tr.marker) {
          tr._origMarkerColor = Array.isArray(tr.marker.color) ? tr.marker.color.slice() : tr.marker.color;
          tr._origMarkerOpacity = (tr.marker.opacity !== undefined) ? tr.marker.opacity : 0.9;
        }
        if (tr.line) {
          tr._origLineColor = tr.line.color;
          tr._origLineOpacity = (tr.line.opacity !== undefined) ? tr.line.opacity : 0.5;
        }
      });
    }
    stashStyles();

    function applyHighlight() {
      const selected = new Set(
        checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value))
      );
      const showAll = selected.size === 0;
      clusterTraces.forEach((traceIdxs, cid) => {
        const active = showAll || selected.has(cid);
        traceIdxs.forEach(idx => {
          const tr = gd.data[idx];
          if (!tr) return;
          if (tr.marker) {
            tr.marker.color = active ? tr._origMarkerColor : 'rgba(160,160,160,0.6)';
            tr.marker.opacity = active ? tr._origMarkerOpacity : 0.2;
          }
          if (tr.line) {
            tr.line.color = active ? tr._origLineColor : 'rgba(180,180,180,0.4)';
            tr.line.opacity = active ? tr._origLineOpacity : 0.25;
          }
          tr.visible = true;
        });
      });
      Plotly.react(gd, gd.data, gd.layout);
    }

    checkboxes.forEach(cb => cb.addEventListener('change', applyHighlight));
  </script>
</body>
</html>
""")
    html = template.substitute(
        checklist_items=checklist_items,
        plot_div=plot_div,
        cluster_traces_json=cluster_traces_json,
    )
    OUTPUT_HTML_INTERACTIVE.write_text(html, encoding="utf-8")
    print(f"Saved interactive highlight map to {OUTPUT_HTML_INTERACTIVE}")

def optimize_k(anchors: np.ndarray, k_range: Iterable[int]) -> Tuple[int, Dict]:
    """
    Find best k using Silhouette Score.
    """
    best_k = None
    best_score = -1.0
    results = {}
    
    print(f"Optimizing k over range {k_range}...")
    
    for k in k_range:
        model = SphericalKMeans(n_clusters=k).fit(anchors)
        labels = model.labels
        
        # Silhouette score (using cosine distance metric)
        # This measures how similar an object is to its own cluster compared to other clusters.
        if len(np.unique(labels)) > 1:
            score = silhouette_score(anchors, labels, metric='cosine')
        else:
            score = -1.0
            
        results[k] = score
        print(f"  k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k, results

def main():
    # 1. Load Data
    all_stars = load_data()
    
    # 2. Filter "Anchor" stars (Bright stars defining the core shape)
    anchors_df = all_stars[all_stars["Vmag"] <= VMAG_ANCHOR_LIMIT].copy()
    anchors_vec = anchors_df[["vx", "vy", "vz"]].values
    
    print(f"Total stars: {len(all_stars)}")
    print(f"Anchor stars (Vmag <= {VMAG_ANCHOR_LIMIT}): {len(anchors_df)}")
    
    # 3. Optimize k (Number of Constellations)
    # Range: 30 to 88 (modern count). 40-60 is usually good for uniform distribution.
    k_candidates = range(35, 65, 3) 
    best_k, _ = optimize_k(anchors_vec, k_candidates)
    print(f"\nSelected optimal k = {best_k} based on Silhouette Score.")
    
    # 4. Final Clustering on Anchors
    final_model = SphericalKMeans(n_clusters=best_k).fit(anchors_vec)
    
    # 5. Partition the Sky: Assign ALL visible stars to these centroids
    # We filter to a slightly dimmer limit for visualization (e.g. 6.5)
    visible_stars = all_stars[all_stars["Vmag"] <= VMAG_VISIBLE_LIMIT].copy()
    visible_vec = visible_stars[["vx", "vy", "vz"]].values
    
    visible_labels = final_model.predict(visible_vec)
    visible_stars["cluster"] = visible_labels
    
    # 6. Visualization
    print("Generating visualization...")
    
    fig = go.Figure()
    cluster_traces: List[List[int]] = [[] for _ in range(best_k)]
    
    # A. Plot Stars
    for cid in range(best_k):
        cluster_data = visible_stars[visible_stars["cluster"] == cid]
        color = COLOR_SEQ[cid % len(COLOR_SEQ)]
        
        # Scatter trace
        fig.add_trace(go.Scatter3d(
            x=cluster_data["vx"], y=cluster_data["vy"], z=cluster_data["vz"],
            mode='markers',
            marker=dict(
                size=np.maximum(1, (7 - cluster_data["Vmag"]) * 1), # Size by brightness
                color=color,
                opacity=0.9,
                line=dict(width=0)
            ),
            name=f"Constellation {cid}",
            text=cluster_data["Label"] + "<br>Vmag: " + cluster_data["Vmag"].astype(str),
            hoverinfo='text'
        ))
        cluster_traces[cid].append(len(fig.data) - 1)
        
        # B. Plot Stick Figures (MST) - Only for brighter stars in the cluster to avoid clutter
        mst_stars = cluster_data[cluster_data["Vmag"] <= VMAG_ANCHOR_LIMIT]
        if len(mst_stars) > 1:
            lx, ly, lz = calculate_mst_lines(mst_stars)
            if lx:
                fig.add_trace(go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode='lines',
                    line=dict(color=color, width=4),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.5
                ))
                cluster_traces[cid].append(len(fig.data) - 1)

        # # C. Plot Centroids (Optional, for reference)
        # cx, cy, cz = final_model.centroids.T
        # fig.add_trace(go.Scatter3d(
        #     x=cx, y=cy, z=cz,
        #     mode='markers',
        #     marker=dict(color='white', size=3, symbol='x'),
        #     name='Centroids'
        # ))

    # Layout adjustments for "Sky View"
    fig.update_layout(
        title=f"Optimized New Constellations (k={best_k})<br>Metric: Spherical Silhouette | Lines: Minimum Spanning Tree",
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='black',
            aspectmode='cube' # Important for sphere shape
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=True,
        legend=dict(itemsizing='constant')
    )
    
    fig.write_html(OUTPUT_HTML)
    save_interactive_highlight_html(fig, cluster_traces)
    print(f"Saved interactive 3D map to {OUTPUT_HTML}")

    # 7. Save each constellation as a standalone RA/Dec PNG with MST outline
    plot_constellation_images(visible_stars, visible_labels, best_k)

if __name__ == "__main__":
    main()
