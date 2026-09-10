"""Aerodynamic force, moment, and dimensionless coefficient calculations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AerodynamicResult:
    """Aerodynamic forces, moments, and dimensionless coefficients."""
    step: int
    time: float
    # Forces
    Fx_total: float
    Fy_total: float
    Fz_total: float
    Fx_pressure: float
    Fy_pressure: float
    Fz_pressure: float
    Fx_viscous: float
    Fy_viscous: float
    Fz_viscous: float
    # Coefficients
    CD: float       # Total drag coefficient (x-axis)
    CD_p: float     # Pressure drag coefficient
    CD_f: float     # Skin friction drag coefficient
    CL: float       # Total lift coefficient (y-axis)
    CL_p: float     # Pressure lift coefficient
    CL_f: float     # Skin friction lift coefficient
    CS: float       # Total side force coefficient (z-axis)
    CS_p: float     # Pressure side force coefficient
    CS_f: float     # Skin friction side force coefficient
    # Moments
    Mx: float       # Rolling moment
    My: float       # Pitching moment
    Mz: float       # Yawing moment
    CMx: float      # Rolling moment coefficient
    CMy: float      # Pitching moment coefficient
    CMz: float      # Yawing moment coefficient
    # Reference quantities
    q_inf: float
    reference_area: float
    reference_length: float

    def summary_line(self) -> str:
        return (
            f"C_D={self.CD:.4f} (p:{self.CD_p:.4f}, f:{self.CD_f:.4f}) "
            f"C_L={self.CL:+.4f} C_S={self.CS:+.4f}"
        )


class AerodynamicForceCalculator:
    """Computes aerodynamic surface forces and moments on designated wall boundaries."""

    def __init__(
        self,
        operator,
        boundary_state=None,
        wall_tag_names: Sequence[str] = ("sphere_wall", "cylinder_wall", "wall", "airfoil_wall"),
        reference_area: Optional[float] = None,
        reference_length: Optional[float] = None,
        moment_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.operator = operator
        self.solver = operator.solver
        self.boundary_state = boundary_state
        self.moment_center = np.asarray(moment_center, dtype=np.float64)

        # Resolve wall boundary tags
        physical_names = self.solver.mesh.physical_names
        self.wall_tags = []
        for name in wall_tag_names:
            if name in physical_names:
                self.wall_tags.append(int(physical_names[name]))

        if not self.wall_tags:
            # Fallback: check if boundary_state has wall tags
            if hasattr(boundary_state, "no_slip_tags"):
                self.wall_tags = [int(t) for t in np.asarray(boundary_state.no_slip_tags)]

        # Precompute face groups and geometric metrics on target wall
        self.wall_faces = self._identify_wall_faces()

        # Geometry dimensions for automatic reference area/length
        xmin, xmax, ymin, ymax, zmin, zmax = self._compute_wall_bounds()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        diameter = max(dx, dy)

        if reference_length is not None:
            self.reference_length = float(reference_length)
        else:
            self.reference_length = float(diameter if diameter > 0 else 1.0)

        if reference_area is not None:
            self.reference_area = float(reference_area)
        else:
            # Check if quasi-2D cylinder extrusion (thin z span)
            if dz < 0.25 * min(dx, dy) and dz > 0:
                # 2D cylinder projected area: D * span
                self.reference_area = float(diameter * dz)
            else:
                # 3D sphere frontal projected area: pi * R^2 = pi * (D/2)^2
                radius = 0.5 * diameter
                self.reference_area = float(np.pi * radius**2)

    def _identify_wall_faces(self):
        wall_faces = []
        for face_type, owner_block_id, neighbor_block_id, indices in self.operator.face_groups:
            if neighbor_block_id >= 0:
                continue
            faces = self.solver.mesh.face_blocks[face_type]
            tags = np.asarray(faces.boundary_tags)[np.asarray(indices)]
            mask = np.isin(tags, self.wall_tags)
            if not np.any(mask):
                continue
            sub_indices = indices[mask]
            mask_indices = np.flatnonzero(mask)
            wall_faces.append({
                "face_type": face_type,
                "owner_block_id": owner_block_id,
                "indices": sub_indices,
                "mask_indices": mask_indices,
            })
        return wall_faces

    def _compute_wall_bounds(self):
        coords = []
        for entry in self.wall_faces:
            face_type = entry["face_type"]
            metrics = self.operator.geometry.face_metrics[face_type]
            pts = np.asarray(metrics.physical_points[entry["indices"]])
            coords.append(pts.reshape(-1, 3))
        if coords:
            all_pts = np.vstack(coords)
            return (
                float(all_pts[:, 0].min()), float(all_pts[:, 0].max()),
                float(all_pts[:, 1].min()), float(all_pts[:, 1].max()),
                float(all_pts[:, 2].min()), float(all_pts[:, 2].max()),
            )
        return (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

    def compute(
        self,
        states: Dict[str, jnp.ndarray],
        q_inf: float,
        step: int = 0,
        current_time: float = 0.0,
    ) -> AerodynamicResult:
        """Compute aerodynamic forces, moments, and non-dimensional coefficients."""
        F_p = np.zeros(3, dtype=np.float64)
        M_p = np.zeros(3, dtype=np.float64)
        F_v = np.zeros(3, dtype=np.float64)
        M_v = np.zeros(3, dtype=np.float64)

        # 1. Pressure forces
        mean_states = self.operator._cell_mean_states(states)
        for entry in self.wall_faces:
            face_type = entry["face_type"]
            owner_block_id = entry["owner_block_id"]
            sub_indices = entry["indices"]
            faces = self.solver.mesh.face_blocks[face_type]
            metrics = self.operator.geometry.face_metrics[face_type]
            owner_block = self.solver.mesh.element_blocks[owner_block_id]
            owner_reference = self.operator.geometry.reference_elements[
                owner_block.element_type
            ]
            owner_cells = jnp.asarray(faces.owner_local)[sub_indices]
            owner_points = metrics.owner_reference_points[sub_indices]
            owner_basis = owner_reference.basis(owner_points)
            owner_anchor = mean_states[owner_block.element_type][owner_cells]
            U_owner = owner_anchor[:, None, :] + jnp.einsum(
                "fqi,fiv->fqv",
                owner_basis,
                states[owner_block.element_type][owner_cells] - owner_anchor[:, None, :],
            )
            W_owner = self.solver.conserved_to_primitive(U_owner)
            p = W_owner[..., 4]  # pressure
            normals = metrics.normals[sub_indices]
            weights = metrics.integration_weights[sub_indices]
            coords = metrics.physical_points[sub_indices]
            r_arm = coords - jnp.asarray(self.moment_center)

            # Pressure traction force: dF_p = p * n * dA
            dF = p[..., None] * normals * weights[..., None]
            dM = jnp.cross(r_arm, dF)

            F_p += np.asarray(jnp.sum(dF, axis=(0, 1)), dtype=np.float64)
            M_p += np.asarray(jnp.sum(dM, axis=(0, 1)), dtype=np.float64)

        # 2. Viscous forces (if Navier-Stokes operator)
        is_viscous = (
            hasattr(self.operator, "_br2_gradients")
            and (self.solver.dynamic_viscosity > 0.0 or self.solver.use_wale)
        )
        if is_viscous:
            _, _, face_data = self.operator._br2_gradients(
                states, self.boundary_state
            )
            for data in face_data:
                if data["neighbor_block_id"] >= 0:
                    continue
                tags = np.asarray(data["faces"].boundary_tags)[np.asarray(data["indices"])]
                mask = np.isin(tags, self.wall_tags)
                if not np.any(mask):
                    continue
                sub_indices = data["indices"][mask]
                sub_mask = np.flatnonzero(mask)
                metrics = data["metrics"]
                normals = metrics.normals[sub_indices]
                weights = metrics.integration_weights[sub_indices]
                coords = metrics.physical_points[sub_indices]
                r_arm = coords - jnp.asarray(self.moment_center)

                owner_grad_basis = self.operator._physical_basis_gradients(
                    data["owner_block_id"], data["owner_cells"][sub_mask],
                    data["owner_points"][sub_mask],
                )
                raw_owner = jnp.einsum(
                    "fim,fqid->fqmd", data["variables_owner_nodes"][sub_mask],
                    owner_grad_basis,
                )
                owner_local_lift = jnp.einsum(
                    "fqi,fimd->fqmd", data["owner_basis"][sub_mask],
                    data["owner_lift"][sub_mask],
                )
                gradient_owner = (
                    raw_owner + self.operator.br2_stabilization * owner_local_lift
                )
                width_owner = self.operator.filter_widths[
                    data["owner_block"].element_type
                ][data["owner_cells"][sub_mask]][:, None]
                flux_owner = self.solver.viscous_physical_flux(
                    data["U_owner"][sub_mask], gradient_owner, width_owner,
                    turbulent_gradient=raw_owner,
                )
                # Momentum viscous flux dotted with outward normal: (tau . n)
                tau_n = jnp.einsum("fqdv,fqd->fqv", flux_owner[..., :, 1:4], normals)
                # Friction force exerted on body is - (tau . n) * dA
                dF_visc = -tau_n * weights[..., None]
                dM_visc = jnp.cross(r_arm, dF_visc)

                F_v += np.asarray(jnp.sum(dF_visc, axis=(0, 1)), dtype=np.float64)
                M_v += np.asarray(jnp.sum(dM_visc, axis=(0, 1)), dtype=np.float64)

        F_total = F_p + F_v
        M_total = M_p + M_v

        # Dimensionless coefficients
        denom_force = max(q_inf * self.reference_area, 1.0e-14)
        denom_moment = max(denom_force * self.reference_length, 1.0e-14)

        CD_p = float(F_p[0] / denom_force)
        CD_f = float(F_v[0] / denom_force)
        CD = CD_p + CD_f

        CL_p = float(F_p[1] / denom_force)
        CL_f = float(F_v[1] / denom_force)
        CL = CL_p + CL_f

        CS_p = float(F_p[2] / denom_force)
        CS_f = float(F_v[2] / denom_force)
        CS = CS_p + CS_f

        CMx = float(M_total[0] / denom_moment)
        CMy = float(M_total[1] / denom_moment)
        CMz = float(M_total[2] / denom_moment)

        return AerodynamicResult(
            step=step,
            time=current_time,
            Fx_total=float(F_total[0]),
            Fy_total=float(F_total[1]),
            Fz_total=float(F_total[2]),
            Fx_pressure=float(F_p[0]),
            Fy_pressure=float(F_p[1]),
            Fz_pressure=float(F_p[2]),
            Fx_viscous=float(F_v[0]),
            Fy_viscous=float(F_v[1]),
            Fz_viscous=float(F_v[2]),
            CD=CD,
            CD_p=CD_p,
            CD_f=CD_f,
            CL=CL,
            CL_p=CL_p,
            CL_f=CL_f,
            CS=CS,
            CS_p=CS_p,
            CS_f=CS_f,
            Mx=float(M_total[0]),
            My=float(M_total[1]),
            Mz=float(M_total[2]),
            CMx=CMx,
            CMy=CMy,
            CMz=CMz,
            q_inf=float(q_inf),
            reference_area=self.reference_area,
            reference_length=self.reference_length,
        )


class AerodynamicHistoryTracker:
    """Records aerodynamic telemetry, streams to CSV, and generates convergence plots."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.history = []
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(self.csv_path, "w") as f:
                f.write(
                    "step,time,Fx_total,Fy_total,Fz_total,"
                    "Fx_pressure,Fy_pressure,Fz_pressure,"
                    "Fx_viscous,Fy_viscous,Fz_viscous,"
                    "CD,CD_p,CD_f,CL,CL_p,CL_f,CS,CS_p,CS_f,"
                    "Mx,My,Mz,CMx,CMy,CMz\n"
                )

    def record(self, result: AerodynamicResult):
        self.history.append(result)
        if self.csv_path is not None:
            with open(self.csv_path, "a") as f:
                f.write(
                    f"{result.step},{result.time:.8f},"
                    f"{result.Fx_total:.8e},{result.Fy_total:.8e},{result.Fz_total:.8e},"
                    f"{result.Fx_pressure:.8e},{result.Fy_pressure:.8e},{result.Fz_pressure:.8e},"
                    f"{result.Fx_viscous:.8e},{result.Fy_viscous:.8e},{result.Fz_viscous:.8e},"
                    f"{result.CD:.6f},{result.CD_p:.6f},{result.CD_f:.6f},"
                    f"{result.CL:.6f},{result.CL_p:.6f},{result.CL_f:.6f},"
                    f"{result.CS:.6f},{result.CS_p:.6f},{result.CS_f:.6f},"
                    f"{result.Mx:.8e},{result.My:.8e},{result.Mz:.8e},"
                    f"{result.CMx:.6f},{result.CMy:.6f},{result.CMz:.6f}\n"
                )

    def save_plot(
        self,
        plot_path: Path,
        mach: Optional[float] = None,
        title_suffix: str = "",
    ):
        """Generate a multi-panel convergence and force breakdown figure."""
        if not self.history:
            return

        steps = [r.step for r in self.history]
        CD = [r.CD for r in self.history]
        CD_p = [r.CD_p for r in self.history]
        CD_f = [r.CD_f for r in self.history]
        CL = [r.CL for r in self.history]
        CS = [r.CS for r in self.history]
        CMy = [r.CMy for r in self.history]
        CMz = [r.CMz for r in self.history]

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), facecolor="#030712", sharex=True
        )

        # 1. Drag Breakdown
        ax1.set_facecolor("#030712")
        ax1.plot(steps, CD, color="#38bdf8", linewidth=2.2, label=f"Total Drag $C_D$ (Latest: {CD[-1]:.4f})")
        ax1.plot(steps, CD_p, color="#f59e0b", linewidth=1.5, linestyle="--", label=f"Pressure Drag $C_{{D,p}}$ ({CD_p[-1]:.4f})")
        ax1.plot(steps, CD_f, color="#10b981", linewidth=1.5, linestyle=":", label=f"Friction Drag $C_{{D,f}}$ ({CD_f[-1]:.4f})")

        # Theoretical reference if applicable
        if mach is not None and abs(mach - 5.0) < 0.5:
            if np.mean(CD[-min(5, len(CD)):]) > 1.1:
                theory_cd = 1.35
                theory_label = "Hypersonic Cylinder Theory ($C_D \\approx 1.35$)"
            else:
                theory_cd = 0.92
                theory_label = "Hypersonic Sphere Theory ($C_D \\approx 0.92$)"
            ax1.axhline(theory_cd, color="#f43f5e", linestyle="-.", linewidth=1.4, alpha=0.85, label=theory_label)

        ax1.set_title("Aerodynamic Drag Coefficient Convergence & Breakdown", color="white", fontsize=12, fontweight="bold", pad=8)
        ax1.set_ylabel("Drag Coefficient $C_D$", color="#94a3b8", fontsize=10)
        ax1.tick_params(colors="#94a3b8")
        ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for s in ax1.spines.values(): s.set_color("#334155")
        ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=9)

        # 2. Lift & Side Force Fluctuations
        ax2.set_facecolor("#030712")
        ax2.plot(steps, CL, color="#ec4899", linewidth=1.8, label=f"Lift $C_L$ (Vertical Y, Latest: {CL[-1]:+.4f})")
        ax2.plot(steps, CS, color="#a855f7", linewidth=1.8, label=f"Side Force $C_S$ (Lateral Z, Latest: {CS[-1]:+.4f})")
        ax2.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.set_title("Unsteady Cross-Stream Aerodynamic Forces", color="white", fontsize=12, fontweight="bold", pad=8)
        ax2.set_ylabel("Force Coefficients ($C_L, C_S$)", color="#94a3b8", fontsize=10)
        ax2.tick_params(colors="#94a3b8")
        ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for s in ax2.spines.values(): s.set_color("#334155")
        ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=9)

        # 3. Aerodynamic Moments
        ax3.set_facecolor("#030712")
        ax3.plot(steps, CMy, color="#06b6d4", linewidth=1.6, label=f"Pitching Moment $C_{{My}}$ ({CMy[-1]:+.4e})")
        ax3.plot(steps, CMz, color="#84cc16", linewidth=1.6, label=f"Yawing Moment $C_{{Mz}}$ ({CMz[-1]:+.4e})")
        ax3.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.6)
        ax3.set_title("Aerodynamic Moments around Reference Center", color="white", fontsize=12, fontweight="bold", pad=8)
        ax3.set_xlabel("Solver Steps", color="#94a3b8", fontsize=10)
        ax3.set_ylabel("Moment Coefficients ($C_M$)", color="#94a3b8", fontsize=10)
        ax3.tick_params(colors="#94a3b8")
        ax3.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        for s in ax3.spines.values(): s.set_color("#334155")
        ax3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=9)

        sup_title = "Aerodynamic Telemetry: Lift, Drag, and Moment History"
        if title_suffix:
            sup_title += f" ({title_suffix})"
        plt.suptitle(sup_title, color="white", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

