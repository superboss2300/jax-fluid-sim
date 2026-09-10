#!/usr/bin/env bash
set -e

echo "============================================================================"
echo "RUNNING SR-71 MACH 2.4 SUPERSONIC SIMULATION (P=2 HIGH-ORDER DG + WALE LES)"
echo "============================================================================"

# 1. Install dependencies
pip install -U "jax[cuda12]" trimesh meshio scipy matplotlib pyevtk gmsh

# 2. Build 600k+ cell wake-refined mesh
python tools/build_sr71_wake_mesh.py
python tools/convert_tet_to_pure_hex.py mesh/sr71_wake_test.msh mesh/sr71_wake_hex.msh

# 3. Run high-order P=2 simulation on RTX 4090
python run_sr71_simulation.py \
    --mesh mesh/sr71_wake_hex.msh \
    --polynomial-degree 2 \
    --wale \
    --maximum-steps 400 \
    --output-directory output/sr71_cinematic

# 4. Generate all 4 long-view MP4 videos
python tools/render_all_cinematic_videos.py

echo "============================================================================"
echo "SUCCESS! All 4 long-view 1080p videos & VTK files ready in output/sr71_cinematic/"
echo "You can now download the videos and STOP your RunPod instance!"
echo "============================================================================"
