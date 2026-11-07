#!/usr/bin/env bash
set -e

# ============================================================================
# JAX FLUID SOLVER — RunPod GPU Showcase
# Two-part simulation: Hypersonic X-43A + 3D Sphere Turbulent Wake
# ============================================================================
#
# Usage:  git clone https://github.com/superboss2300/jax-fluid-sim.git
#         cd jax-fluid-sim && bash run_runpod.sh
#
# Auto-detects GPU VRAM and scales mesh resolution accordingly.
# ============================================================================

echo "============================================================================"
echo "  JAX HIGH-ORDER DG FLUID SOLVER — GPU SHOWCASE"
echo "  Part 1: Hypersonic X-43A Scramjet (Mach 7, bow shock + schlieren)"
echo "  Part 2: 3D Sphere Turbulent Wake (Re=10000, vortex shedding + LES)"
echo "============================================================================"
echo ""

# -------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------
echo ">>> Installing dependencies..."
pip install -q "jax[cuda12]" meshio scipy matplotlib gmsh pyevtk numpy 2>&1 | tail -5
echo "    Dependencies installed."

# -------------------------------------------------------
# 2. Detect GPU and pick mesh resolution
# -------------------------------------------------------
MESH_CONFIG=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits']
    ).decode().strip()
    name, vram_mb = out.split(',')
    vram_gb = int(vram_mb.strip()) / 1024
    print(f'GPU: {name.strip()} ({vram_gb:.0f} GB)', flush=True)

    # Pick sphere mesh preset based on VRAM
    if vram_gb >= 72:
        sphere_preset = 'ultra'
        x43_nwall, x43_nrad, x43_nwake = 100, 140, 140
    elif vram_gb >= 36:
        sphere_preset = 'gpu48gb'
        x43_nwall, x43_nrad, x43_nwake = 80, 100, 100
    elif vram_gb >= 20:
        sphere_preset = 'gpu24gb'
        x43_nwall, x43_nrad, x43_nwake = 60, 80, 80
    else:
        sphere_preset = 'gpu6gb'
        x43_nwall, x43_nrad, x43_nwake = 28, 36, 40

    print(f'SPHERE_PRESET={sphere_preset}')
    print(f'X43_NWALL={x43_nwall}')
    print(f'X43_NRAD={x43_nrad}')
    print(f'X43_NWAKE={x43_nwake}')
except Exception as e:
    print(f'WARNING: GPU detection failed ({e}), using conservative defaults')
    print('SPHERE_PRESET=gpu6gb')
    print('X43_NWALL=28')
    print('X43_NRAD=36')
    print('X43_NWAKE=40')
")

echo "$MESH_CONFIG"

# Parse config values
SPHERE_PRESET=$(echo "$MESH_CONFIG" | grep SPHERE_PRESET | cut -d= -f2)
X43_NWALL=$(echo "$MESH_CONFIG" | grep X43_NWALL | cut -d= -f2)
X43_NRAD=$(echo "$MESH_CONFIG" | grep X43_NRAD | cut -d= -f2)
X43_NWAKE=$(echo "$MESH_CONFIG" | grep X43_NWAKE | cut -d= -f2)

echo ""
echo "Mesh resolution: sphere=$SPHERE_PRESET, x43a=${X43_NWALL}x${X43_NRAD}x${X43_NWAKE}"

# Verify JAX sees the GPU
python3 -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
if not any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices):
    print('ERROR: JAX cannot see GPU! Check CUDA installation.')
    exit(1)
print('JAX GPU OK')
"

# -------------------------------------------------------
# 3. PART 1: Hypersonic X-43A Scramjet Waverider
# -------------------------------------------------------
echo ""
echo "============================================================================"
echo "  PART 1: X-43A SCRAMJET — MACH 7, alpha=12deg, Re=100,000"
echo "  Structured hex27 C-grid: ${X43_NWALL}x${X43_NRAD}x${X43_NWAKE}"
echo "  Features: Bow shock, expansion fan, WENO subcell FV, WALE LES"
echo "  Renders: Mach field, temperature, vorticity, numerical schlieren"
echo "============================================================================"

echo ">>> Generating X-43A structured C-grid..."
python3 tools/generate_hypersonic_airfoil_grid.py \
    --profile x43a \
    --alpha 12 \
    --n_wall "$X43_NWALL" \
    --n_rad "$X43_NRAD" \
    --n_wake "$X43_NWAKE" \
    --output mesh/x43a_runpod.msh

echo ">>> Running hypersonic X-43A simulation..."
python3 run_hypersonic_airfoil.py \
    --mesh mesh/x43a_runpod.msh \
    --mach 7.0 \
    --alpha 12.0 \
    --reynolds 100000 \
    --polynomial-degree 2 \
    --cfl 0.2 \
    --maximum-steps 3000 \
    --target-time 0.5 \
    --progress-updates 100 \
    --checkpoint-interval 500 \
    --render-interval 100 \
    --wale \
    --subcell-fv \
    --reconstruction weno \
    --output-directory output/x43a_mach7

echo ">>> Part 1 complete! Snapshots in output/x43a_mach7/"

# -------------------------------------------------------
# 4. PART 2: 3D Sphere Turbulent Wake
# -------------------------------------------------------
echo ""
echo "============================================================================"
echo "  PART 2: 3D SPHERE WAKE — Mach 0.3, Re=10,000, preset=$SPHERE_PRESET"
echo "  Features: Von Karman vortex shedding, turbulent separation, WALE LES"
echo "  Output: VTK frames for ParaView Q-criterion animation"
echo "============================================================================"

echo ">>> Generating sphere tunnel mesh (preset: $SPHERE_PRESET)..."
python3 tools/generate_sphere_tunnel.py --preset "$SPHERE_PRESET"

echo ">>> Running 3D sphere turbulence simulation..."
python3 run_dg_sphere.py \
    --mesh "mesh/sphere_tunnel_${SPHERE_PRESET}.msh" \
    --mach 0.3 \
    --reynolds 10000 \
    --wale \
    --polynomial-degree 2 \
    --vtk \
    --maximum-steps 5000 \
    --target-time 0.015 \
    --progress-updates 100 \
    --output-directory output/sphere_turbulence

echo ">>> Part 2 complete! VTK frames in output/sphere_turbulence/"

# -------------------------------------------------------
# 5. Package outputs for download
# -------------------------------------------------------
echo ""
echo ">>> Packaging all outputs..."
tar -czf runpod_results.tar.gz \
    output/x43a_mach7/ \
    output/sphere_turbulence/ \
    2>/dev/null || true

RESULT_SIZE=$(du -sh runpod_results.tar.gz 2>/dev/null | cut -f1)

echo ""
echo "============================================================================"
echo "  ALL SIMULATIONS COMPLETE!"
echo ""
echo "  Results:"
echo "    output/x43a_mach7/         — Hypersonic schlieren snapshots + checkpoint"
echo "    output/sphere_turbulence/  — VTK animation frames + center-slice PNG"
echo "    runpod_results.tar.gz      — All outputs packaged ($RESULT_SIZE)"
echo ""
echo "  Download runpod_results.tar.gz and STOP your RunPod instance!"
echo "============================================================================"
