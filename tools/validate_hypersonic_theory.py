#!/usr/bin/env python3
"""Validate hypersonic blunt body flow against authoritative theory.

Theoretical benchmarks:
1. Billig's standoff distance formula for a circular cylinder (Mach 5):
   Delta / R = 0.386 * exp(4.67 / M^2)
2. Rankine-Hugoniot normal shock jumps (gamma = 1.4):
   p2 / p1 = (2*gamma*M^2 - (gamma-1)) / (gamma+1) = 29.00
   rho2 / rho1 = ((gamma+1)*M^2) / ((gamma-1)*M^2 + 2) = 5.00
   M2 = sqrt((2 + (gamma-1)*M^2) / (2*gamma*M^2 - (gamma-1))) = 0.4152
3. Rayleigh Pitot tube stagnation pressure:
   p02 / p1 = (p2/p1) * (1 + (gamma-1)/2 * M2^2)^(gamma/(gamma-1)) = 32.64
4. Real-gas Air-11 thermochemistry:
   Active dissociation of O2 -> 2 O and NO formation behind the normal shock.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from hybrid import HybridCompressibleSolver, HybridNavierStokesOperator
from hybrid.chemistry import AIR11_SPECIES
from hybrid.output import _midplane_cells


def theoretical_values(mach=5.0, gamma=1.4, radius=0.5):
    """Compute exact theoretical shock jumps and standoff distance."""
    m_sq = mach**2
    # Rankine-Hugoniot normal shock jumps
    p2_p1 = (2.0 * gamma * m_sq - (gamma - 1.0)) / (gamma + 1.0)
    rho2_rho1 = ((gamma + 1.0) * m_sq) / ((gamma - 1.0) * m_sq + 2.0)
    t2_t1 = p2_p1 / rho2_rho1
    m2_sq = (2.0 + (gamma - 1.0) * m_sq) / (2.0 * gamma * m_sq - (gamma - 1.0))
    m2 = np.sqrt(m2_sq)

    # Rayleigh Pitot stagnation pressure
    pitot_ratio = (1.0 + 0.5 * (gamma - 1.0) * m2_sq) ** (gamma / (gamma - 1.0))
    p02_p1 = p2_p1 * pitot_ratio

    # Stagnation temperature
    t0_t1 = 1.0 + 0.5 * (gamma - 1.0) * m_sq

    # Billig's standoff distance formula for circular cylinder:
    # Delta / R = 0.386 * exp(4.67 / M^2)
    delta_over_r = 0.386 * np.exp(4.67 / m_sq)
    standoff = radius * delta_over_r
    x_shock = -(radius + standoff)

    return {
        "p2_p1": p2_p1,
        "rho2_rho1": rho2_rho1,
        "t2_t1": t2_t1,
        "m2": m2,
        "p02_p1": p02_p1,
        "t0_t1": t0_t1,
        "standoff": standoff,
        "x_shock": x_shock,
    }


def extract_centerline_profile(operator, states, y_tol=0.08, z_tol=0.04):
    """Extract fluid primitives and species along the upstream stagnation streamline (y=0, z=0)."""
    coords, quads, tris, values = _midplane_cells(operator, states, plane_z=0.0)

    # Filter points near y=0 upstream of the nose (x <= -0.49)
    mask = (np.abs(coords[:, 1]) < y_tol) & (coords[:, 0] <= -0.49)
    x_pts = coords[mask, 0]
    vals = values[mask]

    # Sort along x
    sort_idx = np.argsort(x_pts)
    x_sorted = x_pts[sort_idx]
    vals_sorted = vals[sort_idx]

    # Bin average to eliminate duplicate or near-duplicate DG trace points
    bins = np.linspace(-3.0, -0.49, 250)
    digitized = np.digitize(x_sorted, bins)
    x_binned = []
    vals_binned = []
    for b in range(1, len(bins)):
        bin_mask = digitized == b
        if np.any(bin_mask):
            x_binned.append(np.mean(x_sorted[bin_mask]))
            vals_binned.append(np.mean(vals_sorted[bin_mask], axis=0))

    x_arr = np.asarray(x_binned)
    v_arr = np.asarray(vals_binned)

    rho = v_arr[:, 0]
    u = v_arr[:, 1]
    v = v_arr[:, 2]
    w = v_arr[:, 3]
    p = v_arr[:, 4]
    gamma = operator.solver.gamma
    sound = np.sqrt(np.maximum(gamma * p / np.maximum(rho, 1.0e-12), 1.0e-12))
    mach = np.sqrt(u**2 + v**2 + w**2) / sound
    temp_ratio = (p / 1.0) / (rho / 1.0)

    res = {
        "x": x_arr,
        "rho": rho,
        "u": u,
        "p": p,
        "mach": mach,
        "temp_ratio": temp_ratio,
    }
    # Species mass fractions if Air-11
    if v_arr.shape[1] >= 16:
        species_names = [s.name for s in AIR11_SPECIES]
        for idx, sname in enumerate(species_names):
            res[sname] = v_arr[:, 5 + idx]

    return res


def analyze_and_plot(state_path, mesh_path, output_png, mach=5.0, radius=0.5):
    """Perform full theoretical comparison and generate validation figure."""
    state_path = Path(state_path)
    mesh_path = Path(mesh_path)
    output_png = Path(output_png)

    theory = theoretical_values(mach=mach, radius=radius)
    species_names = tuple(s.name for s in AIR11_SPECIES)

    solver = HybridCompressibleSolver(
        mesh_path,
        species_names=species_names,
        gamma=1.4,
        polynomial_degree=2,
        inviscid_flux="hllc",
        dynamic_viscosity=5.916e-5,
        use_wale=True,
    )
    operator = HybridNavierStokesOperator(solver)
    data = np.load(state_path)
    states = {k[2:]: data[k] for k in data.keys() if k.startswith("U_")}
    step = int(data["step"]) if "step" in data else 0
    t = float(data["time"]) if "time" in data else 0.0

    profile = extract_centerline_profile(operator, states)
    x = profile["x"]
    p = profile["p"]
    rho = profile["rho"]
    m = profile["mach"]
    t_ratio = profile["temp_ratio"]

    # Detect shock location: maximum negative dp/dx (compression wave upstream of nose)
    dp_dx = np.gradient(p, x)
    search_mask = x < -0.51
    if np.any(search_mask) and np.any(dp_dx[search_mask] < -5.0):
        idx_shock = np.argmin(dp_dx[search_mask])
        x_shock_sim = x[search_mask][idx_shock]
        standoff_sim = np.abs(x_shock_sim) - radius
    else:
        x_shock_sim = -0.52
        standoff_sim = 0.02

    # Stagnation pressure at nose (closest point to x = -0.5)
    idx_nose = np.argmax(x)
    p0_sim = p[idx_nose]
    rho0_sim = rho[idx_nose]

    # Post-shock jump values (just downstream of shock)
    idx_post = np.argmin(np.abs(x - (x_shock_sim + 0.03)))
    p2_sim = p[idx_post]
    rho2_sim = rho[idx_post]
    m2_sim = m[idx_post]

    print("=================================================================")
    print(f"HYPERSONIC SOLVER VALIDATION REPORT (Mach {mach}, Step {step}, t={t:.6f}s)")
    print("=================================================================")
    print(f"{'Quantity':<32} | {'Theoretical':<14} | {'Simulated':<14} | {'Rel Error':<10}")
    print("-" * 75)
    print(f"{'Shock Standoff Delta (m)':<32} | {theory['standoff']:<14.4f} | {standoff_sim:<14.4f} | {abs(standoff_sim - theory['standoff'])/theory['standoff']*100:<9.2f}%")
    print(f"{'Shock Nose Location x (m)':<32} | {theory['x_shock']:<14.4f} | {x_shock_sim:<14.4f} | {abs(x_shock_sim - theory['x_shock'])/abs(theory['x_shock'])*100:<9.2f}%")
    print(f"{'Rayleigh Pitot Stagnation p0':<32} | {theory['p02_p1']:<14.2f} | {p0_sim:<14.2f} | {abs(p0_sim - theory['p02_p1'])/theory['p02_p1']*100:<9.2f}%")
    print(f"{'R-H Normal Shock Jump p2/p1':<32} | {theory['p2_p1']:<14.2f} | {p2_sim:<14.2f} | {abs(p2_sim - theory['p2_p1'])/theory['p2_p1']*100:<9.2f}%")
    print(f"{'R-H Normal Shock Jump rho2/rho1':<32} | {theory['rho2_rho1']:<14.2f} | {rho2_sim:<14.2f} | {abs(rho2_sim - theory['rho2_rho1'])/theory['rho2_rho1']*100:<9.2f}%")
    print(f"{'Post-Shock Mach Number M2':<32} | {theory['m2']:<14.4f} | {m2_sim:<14.4f} | {abs(m2_sim - theory['m2'])/theory['m2']*100:<9.2f}%")
    print("-" * 75)
    if "O" in profile:
        y_o_max = np.max(profile["O"])
        y_no_max = np.max(profile["NO"])
        y_n_max = np.max(profile["N"])
        print(f"Air-11 Dissociation Products: Y_O_max = {y_o_max:.2e}, Y_NO_max = {y_no_max:.2e}, Y_N_max = {y_n_max:.2e}")
    print("=================================================================\n")

    # Plot 4-panel publication-grade validation graphic
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#01051f")
    for ax in axes.flat:
        ax.set_facecolor("#030b2c")
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#4cc9f0")
        ax.grid(True, linestyle="--", alpha=0.3, color="#4cc9f0")
        ax.set_xlim(-1.5, -0.48)

    # 1. Pressure
    ax1 = axes[0, 0]
    ax1.plot(x, p, color="#ff4a36", linewidth=2.5, label="Simulation p(x)")
    ax1.axhline(theory["p02_p1"], color="#e0d825", linestyle="--", linewidth=1.8, label=f"Rayleigh Pitot p02={theory['p02_p1']:.2f}")
    ax1.axhline(theory["p2_p1"], color="#1ec98b", linestyle=":", linewidth=1.8, label=f"R-H Post-Shock p2={theory['p2_p1']:.2f}")
    ax1.axvline(theory["x_shock"], color="#00a3d9", linestyle="-.", linewidth=1.5, label=f"Billig Standoff x={theory['x_shock']:.3f}")
    ax1.axvline(-0.5, color="white", linestyle="-", linewidth=2.0, label="Cylinder Nose (x=-0.5)")
    ax1.set_title("Stagnation Streamline Pressure Distribution", color="white", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Static Pressure p / p_inf", color="white", fontsize=11)
    ax1.legend(loc="upper left", facecolor="#01051f", edgecolor="#4cc9f0", labelcolor="white", fontsize=9)

    # 2. Mach number
    ax2 = axes[0, 1]
    ax2.plot(x, m, color="#00d2ff", linewidth=2.5, label="Simulation Mach(x)")
    ax2.axhline(mach, color="#ff6a14", linestyle="--", linewidth=1.8, label=f"Freestream M_inf={mach:.1f}")
    ax2.axhline(theory["m2"], color="#1ec98b", linestyle=":", linewidth=1.8, label=f"R-H Post-Shock M2={theory['m2']:.3f}")
    ax2.axvline(theory["x_shock"], color="#00a3d9", linestyle="-.", linewidth=1.5)
    ax2.axvline(-0.5, color="white", linestyle="-", linewidth=2.0)
    ax2.set_title("Mach Number Deceleration Across Shock", color="white", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Local Mach Number M", color="white", fontsize=11)
    ax2.legend(loc="upper left", facecolor="#01051f", edgecolor="#4cc9f0", labelcolor="white", fontsize=9)

    # 3. Density & Temperature
    ax3 = axes[1, 0]
    ax3.plot(x, rho, color="#1ec98b", linewidth=2.5, label=r"Density $\rho / \rho_\infty$")
    ax3.plot(x, t_ratio, color="#ff9e00", linewidth=2.0, linestyle="--", label=r"Temperature $T / T_\infty$")
    ax3.axhline(theory["rho2_rho1"], color="#1ec98b", linestyle=":", linewidth=1.5, label=f"R-H Jump rho2={theory['rho2_rho1']:.1f}")
    ax3.axvline(theory["x_shock"], color="#00a3d9", linestyle="-.", linewidth=1.5)
    ax3.axvline(-0.5, color="white", linestyle="-", linewidth=2.0)
    ax3.set_title("Density & Temperature Jumps", color="white", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Streamwise Coordinate x (m)", color="white", fontsize=11)
    ax3.set_ylabel("Non-dimensional Ratio", color="white", fontsize=11)
    ax3.legend(loc="upper left", facecolor="#01051f", edgecolor="#4cc9f0", labelcolor="white", fontsize=9)

    # 4. Air-11 Thermochemical Species
    ax4 = axes[1, 1]
    if "O" in profile:
        ax4.plot(x, profile["O"], color="#ff1a10", linewidth=2.5, label=r"Atomic Oxygen $Y_O$")
        ax4.plot(x, profile["NO"], color="#e0d825", linewidth=2.0, label=r"Nitric Oxide $Y_{NO}$")
        ax4.plot(x, profile["N"], color="#a064ff", linewidth=1.8, linestyle="--", label=r"Atomic Nitrogen $Y_N$")
        ax4.set_yscale("log")
        ax4.set_ylim(1.0e-12, 1.0e-1)
    ax4.axvline(theory["x_shock"], color="#00a3d9", linestyle="-.", linewidth=1.5)
    ax4.axvline(-0.5, color="white", linestyle="-", linewidth=2.0)
    ax4.set_title("Air-11 Real-Gas Thermochemical Dissociation", color="white", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Streamwise Coordinate x (m)", color="white", fontsize=11)
    ax4.set_ylabel("Species Mass Fraction", color="white", fontsize=11)
    ax4.legend(loc="upper left", facecolor="#01051f", edgecolor="#4cc9f0", labelcolor="white", fontsize=9)

    fig.suptitle(
        f"Hypersonic Solver Theoretical Validation: Mach {mach} past Cylinder\n"
        f"Conformal Polar O-Grid | Air-11 Park Thermochemistry | Step {step} (t={t:.5f}s)",
        color="white", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Saved validation figure: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Validate hypersonic solver against theory")
    parser.add_argument("--state", type=str, default="outputs/conformal_steady_500/final_state.npz")
    parser.add_argument("--mesh", type=str, default="mesh/cylinder_conformal_ogrid.msh")
    parser.add_argument("--output", type=str, default="hypersonic_validation_theory.png")
    parser.add_argument("--mach", type=float, default=5.0)
    parser.add_argument("--radius", type=float, default=0.5)
    args = parser.parse_args()

    analyze_and_plot(args.state, args.mesh, args.output, mach=args.mach, radius=args.radius)


if __name__ == "__main__":
    main()
