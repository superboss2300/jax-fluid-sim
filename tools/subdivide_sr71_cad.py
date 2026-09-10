"""Subdivide long edges in SR-71 CAD surface to prevent sliver hexes.

Eliminates the 10.5m wing edges, reducing maximum edge length to <= 1.2m
while preserving watertight topology, exact volume, and machine-level symmetry.
"""

from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def subdivide_sr71_stl(
    input_stl: str = "mesh/sr71_perfect_clean.stl",
    output_stl: str = "mesh/sr71_faceted_clean.stl",
    max_edge: float = 1.2,
):
    print(f"Loading {input_stl}...")
    m = trimesh.load(input_stl)
    print(f"Original: {len(m.vertices)} verts, {len(m.faces)} faces, max edge: {m.edges_unique_length.max():.3f} m")

    v_sub, f_sub = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=max_edge)
    m_sub = trimesh.Trimesh(vertices=v_sub, faces=f_sub)

    print(f"Subdivided: {len(m_sub.vertices)} verts, {len(m_sub.faces)} faces, max edge: {m_sub.edges_unique_length.max():.3f} m, min edge: {m_sub.edges_unique_length.min():.5f} m")
    print(f"Watertight: {m_sub.is_watertight}, Volume: {m_sub.volume:.4f} m^3")

    # Verify symmetry
    v = m_sub.vertices
    v_mirror = v.copy()
    v_mirror[:, 2] = -v_mirror[:, 2]
    tree = cKDTree(v)
    dists, _ = tree.query(v_mirror)
    print(f"Symmetry check: max distance = {dists.max():.6e} m")

    # Export STL
    Path(output_stl).parent.mkdir(parents=True, exist_ok=True)
    m_sub.export(output_stl)
    print(f"Saved pristine subdivided STL to {output_stl}")

if __name__ == "__main__":
    subdivide_sr71_stl()
