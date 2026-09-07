"""Test discrete conservation of hybrid DG / subcell FV coupling."""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid import (
    FarfieldWallBoundary,
    HybridCompressibleSolver,
    HybridNavierStokesOperator,
    HybridMOODSubcellLimiter,
)
from hybrid.subcell import HybridSubcellFV


def element_integrals(states, operator):
    """Compute volume integrals of conserved variables for each element."""
    integrals = {}
    for block_id, block in enumerate(operator.mesh.element_blocks):
        name = block.element_type
        reference = operator.geometry.reference_elements[name]
        metrics = operator.geometry.element_metrics[block_id]
        U = states[name]
        # Evaluate at volume quadrature points
        U_q = jnp.einsum(
            "qi,eiv->eqv",
            reference.basis(reference.volume_points),
            U,
            precision=jax.lax.Precision.HIGHEST,
        )
        # Integrate with physical quadrature weights
        int_U = jnp.einsum(
            "eqv,eq->ev",
            U_q,
            metrics.integration_weights,
            precision=jax.lax.Precision.HIGHEST,
        )
        integrals[name] = int_U
    return integrals


def total_domain_conserved(states, operator):
    integrals = element_integrals(states, operator)
    return sum(jnp.sum(val, axis=0) for val in integrals.values())


def test_conservation():
    mesh_path = "mesh/sphere_tunnel_hybrid.msh"
    solver = HybridCompressibleSolver(mesh_path, polynomial_degree=2, dynamic_viscosity=0.0)
    operator = HybridNavierStokesOperator(solver)
    farfield = jnp.asarray([1.0, 2.0, 0.0, 0.0, 1.0])
    boundary = FarfieldWallBoundary(solver, farfield)

    # Initial state
    states = solver.uniform_state_blocks(1.0, jnp.asarray([2.0, 0.0, 0.0]), 1.0)
    dt = 1.0e-5

    # Compute DG candidate
    inviscid = operator.inviscid_rhs(states, boundary)
    dg_candidate = {k: states[k] + dt * inviscid[k] for k in states}

    # Mark some elements as troubled (e.g. elements 10, 11, 50 in tetra, and 5 in wedge)
    troubled = {
        "tetra": jnp.zeros(len(states["tetra"]), dtype=bool).at[jnp.asarray([10, 11, 50, 100])].set(True),
        "wedge": jnp.zeros(len(states["wedge"]), dtype=bool).at[jnp.asarray([5, 12])].set(True),
    }

    for recon in ["constant", "tvd", "weno"]:
        subcell = HybridSubcellFV(operator, reconstruction=recon)
        corrected = subcell.recompute(states, dg_candidate, dt, troubled, boundary_state=boundary)

        total_dg = total_domain_conserved(dg_candidate, operator)
        total_subcell = total_domain_conserved(corrected, operator)

        diff = jnp.max(jnp.abs(total_subcell - total_dg))
        print(f"[{recon.upper()}] Max discrepancy in total conserved quantities: {float(diff):.6e}")
        assert float(diff) < 1.0e-3, f"Conservation violation ({recon}): {float(diff)}"

    print("All discrete conservation tests (constant, TVD & WENO) verified successfully!")


if __name__ == "__main__":
    test_conservation()
