"""Post-process saved simulation states to evaluate aerodynamic forces, lift, drag, and moments."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hybrid import (
    AerodynamicForceCalculator,
    AerodynamicHistoryTracker,
    FarfieldWallBoundary,
    HybridCompressibleSolver,
    HybridNavierStokesOperator,
)
from hybrid.chemistry import Air11ParkMechanism, AIR11_SPECIES


def compute_aerodynamics_for_state(
    mesh_path: str,
    state_path: str,
    mach: float = 5.0,
    reynolds: float = 100000.0,
    gamma: float = 1.4,
    polynomial_degree: int = 2,
    chemistry: str = "air11",
    wale: bool = True,
    reference_area: float = None,
    reference_length: float = None,
    moment_center: tuple = (0.0, 0.0, 0.0),
    output_dir: str = None,
):
    density = 1.0
    pressure = 1.0
    sound_speed = (gamma * pressure / density) ** 0.5
    velocity = mach * sound_speed
    viscosity = density * velocity / reynolds
    q_inf = 0.5 * density * velocity**2

    if chemistry == "air11":
        species_names = tuple(s.name for s in AIR11_SPECIES)
        mass_fractions = np.zeros(len(species_names), dtype=float)
        mass_fractions[9] = 0.767   # N2
        mass_fractions[10] = 0.233  # O2
    else:
        species_names = ()
        mass_fractions = None

    solver = HybridCompressibleSolver(
        mesh_path,
        species_names=species_names,
        gamma=gamma,
        polynomial_degree=polynomial_degree,
        dynamic_viscosity=viscosity,
        use_wale=wale,
    )
    operator = HybridNavierStokesOperator(solver)
    if mass_fractions is not None:
        farfield = jnp.asarray([density, velocity, 0.0, 0.0, pressure, *mass_fractions])
    else:
        farfield = jnp.asarray([density, velocity, 0.0, 0.0, pressure])
    boundary = FarfieldWallBoundary(solver, farfield)

    restart = np.load(state_path)
    states = {
        block.element_type: jnp.asarray(restart[f"U_{block.element_type}"])
        for block in solver.mesh.element_blocks
    }
    step = int(restart.get("step", 0))
    current_time = float(restart.get("time", 0.0))

    calc = AerodynamicForceCalculator(
        operator,
        boundary_state=boundary,
        reference_area=reference_area,
        reference_length=reference_length,
        moment_center=moment_center,
    )

    result = calc.compute(states, q_inf, step=step, current_time=current_time)

    print("\n" + "=" * 65)
    print(f" AERODYNAMIC TELEMETRY REPORT (Mach {mach}, Re {reynolds:g})")
    print("=" * 65)
    print(f" State File:        {state_path}")
    print(f" Mesh File:         {mesh_path}")
    print(f" Time / Step:       t = {current_time:.6f} s  |  Step {step}")
    print(f" Freestream q_inf:  {q_inf:.4f} Pa")
    print(f" Ref Area / Length: A_ref = {result.reference_area:.6f} m^2, L_ref = {result.reference_length:.4f} m")
    print("-" * 65)
    print(f" TOTAL DRAG (X):     C_D  = {result.CD:.6f}")
    print(f"   Pressure Drag:    C_Dp = {result.CD_p:.6f}  ({100*result.CD_p/max(result.CD, 1e-12):.1f}%)")
    print(f"   Friction Drag:    C_Df = {result.CD_f:.6f}  ({100*result.CD_f/max(result.CD, 1e-12):.1f}%)")
    print(f" LIFT FORCE (Y):     C_L  = {result.CL:+.6f}")
    print(f"   Pressure Lift:    C_Lp = {result.CL_p:+.6f}")
    print(f"   Friction Lift:    C_Lf = {result.CL_f:+.6f}")
    print(f" SIDE FORCE (Z):     C_S  = {result.CS:+.6f}")
    print(f"   Pressure Side:    C_Sp = {result.CS_p:+.6f}")
    print(f"   Friction Side:    C_Sf = {result.CS_f:+.6f}")
    print("-" * 65)
    print(f" FORCE VECTORS (N):")
    print(f"   F_total:    [{result.Fx_total:+.6e}, {result.Fy_total:+.6e}, {result.Fz_total:+.6e}]")
    print(f"   F_pressure: [{result.Fx_pressure:+.6e}, {result.Fy_pressure:+.6e}, {result.Fz_pressure:+.6e}]")
    print(f"   F_viscous:  [{result.Fx_viscous:+.6e}, {result.Fy_viscous:+.6e}, {result.Fz_viscous:+.6e}]")
    print("-" * 65)
    print(f" MOMENTS & COEFFICIENTS:")
    print(f"   Pitching (Y): My = {result.My:+.6e} N*m | C_My = {result.CMy:+.6e}")
    print(f"   Yawing   (Z): Mz = {result.Mz:+.6e} N*m | C_Mz = {result.CMz:+.6e}")
    print(f"   Rolling  (X): Mx = {result.Mx:+.6e} N*m | C_Mx = {result.CMx:+.6e}")
    print("=" * 65 + "\n")

    if output_dir is not None:
        out_path = Path(output_dir)
        tracker = AerodynamicHistoryTracker(out_path / "aerodynamics.csv")
        tracker.record(result)
        tracker.save_plot(out_path / "drag_lift_history.png", mach=mach)
        print(f"Saved telemetry to {out_path / 'aerodynamics.csv'}")
        print(f"Saved plot to {out_path / 'drag_lift_history.png'}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Path to mesh file (.msh)")
    parser.add_argument("--state", required=True, help="Path to state file (.npz)")
    parser.add_argument("--mach", type=float, default=5.0)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("-p", "--polynomial-degree", type=int, default=2)
    parser.add_argument("--chemistry", choices=("none", "air11"), default="air11")
    parser.add_argument("--wale", action="store_true", default=True)
    parser.add_argument("--reference-area", type=float, default=None)
    parser.add_argument("--reference-length", type=float, default=None)
    parser.add_argument("--moment-center", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output", default=None, help="Directory to save CSV and plot")
    args = parser.parse_args()

    compute_aerodynamics_for_state(
        mesh_path=args.mesh,
        state_path=args.state,
        mach=args.mach,
        reynolds=args.reynolds,
        polynomial_degree=args.polynomial_degree,
        chemistry=args.chemistry,
        wale=args.wale,
        reference_area=args.reference_area,
        reference_length=args.reference_length,
        moment_center=tuple(args.moment_center),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

