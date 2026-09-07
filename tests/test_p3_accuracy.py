"""Verification of higher-order P=3 DG reference elements and spatial operators."""

import jax.numpy as jnp
import numpy as np

from hybrid.elements import make_reference_element
from hybrid.mesh import HybridMesh
from hybrid.geometry import HybridGeometry
from hybrid.solver import HybridCompressibleSolver
from hybrid.spatial import HybridEulerOperator


def test_p3_reference_elements():
    """Verify Kronecker delta, partition of unity, and mass matrices for P=3."""
    for elem_type, expected_nodes in [("tetra", 20), ("wedge", 40), ("hexahedron", 64)]:
        ref = make_reference_element(elem_type, polynomial_degree=3)
        assert ref.polynomial_degree == 3
        assert ref.nodes.shape == (expected_nodes, 3)

        # 1. Kronecker delta property at nodes
        basis_at_nodes = ref.basis(ref.nodes)
        eye = jnp.eye(expected_nodes)
        delta_err = float(jnp.max(jnp.abs(basis_at_nodes - eye)))
        print(f"{elem_type} P=3 node interpolation error: {delta_err:.4e}")
        assert delta_err < 1.0e-12

        # 2. Partition of unity across all volume quadrature points
        basis_at_quad = ref.basis(ref.volume_points)
        sum_basis = jnp.sum(basis_at_quad, axis=-1)
        unity_err = float(jnp.max(jnp.abs(sum_basis - 1.0)))
        print(f"{elem_type} P=3 partition of unity error: {unity_err:.4e}")
        assert unity_err < 1.0e-12

        # 3. Mass matrix symmetry and positive definiteness
        M = ref.mass_matrix
        M_inv = ref.inverse_mass_matrix
        assert float(jnp.max(jnp.abs(M - M.T))) < 1.0e-14
        eigs = jnp.linalg.eigvalsh(M)
        assert jnp.all(eigs > 0.0)
        cond = float(jnp.max(eigs) / jnp.min(eigs))
        print(f"{elem_type} P=3 mass condition number: {cond:.2f}")

        # Mass matrix inverse roundtrip
        ident_err = float(jnp.max(jnp.abs(jnp.dot(M, M_inv) - eye)))
        assert ident_err < 1.0e-11


def test_p3_exact_polynomial_integration():
    """Verify that volume integration of cubic polynomials is exact to machine precision."""
    ref = make_reference_element("tetra", polynomial_degree=3)
    # Define a test cubic polynomial: f(x, y, z) = 1 + 2x - 3y + 4z + 5x^2 - 6xy + 7z^3
    def f_poly(x, y, z):
        return 1.0 + 2.0*x - 3.0*y + 4.0*z + 5.0*(x**2) - 6.0*x*y + 7.0*(z**3)

    # Analytic integral over reference tetrahedron {(x,y,z) >= 0, x+y+z <= 1}:
    # Int(1) = 1/6
    # Int(x) = Int(y) = Int(z) = 1/24
    # Int(x^2) = 1/60
    # Int(xy) = 1/120
    # Int(z^3) = 1/120
    expected_integral = (
        1.0 * (1.0 / 6.0)
        + 2.0 * (1.0 / 24.0)
        - 3.0 * (1.0 / 24.0)
        + 4.0 * (1.0 / 24.0)
        + 5.0 * (1.0 / 60.0)
        - 6.0 * (1.0 / 120.0)
        + 7.0 * (1.0 / 120.0)
    )

    quad_pts = ref.volume_points
    f_at_quad = f_poly(quad_pts[:, 0], quad_pts[:, 1], quad_pts[:, 2])
    numerical_integral = float(jnp.sum(ref.volume_weights * f_at_quad))

    abs_err = abs(numerical_integral - expected_integral)
    print(f"P=3 exact polynomial integration error: {abs_err:.4e}")
    assert abs_err < 1.0e-14


def test_p3_hybrid_spatial_operator():
    """Verify HybridCompressibleSolver and HybridEulerOperator with P=3 on mixed mesh."""
    mesh_path = "mesh/hybrid_smoke.msh"
    solver = HybridCompressibleSolver(
        mesh_path=mesh_path,
        polynomial_degree=3,
        gamma=1.4,
        gas_constant=1.0,
    )
    geom = HybridGeometry(solver.mesh, polynomial_degree=3)
    op = HybridEulerOperator(solver)

    # Initial uniform state
    states = {}
    for block in solver.mesh.element_blocks:
        num_cells = block.connectivity.shape[0]
        ref = solver.reference_elements[block.element_type]
        num_nodes = ref.nodes.shape[0]
        # Uniform Mach 2.0 flow: [rho, rhou, rhov, rhow, E]
        U0 = jnp.array([1.0, 2.0, 0.0, 0.0, 3.5], dtype=jnp.float64)
        states[block.element_type] = jnp.broadcast_to(
            U0, (num_cells, num_nodes, 5)
        )

    # Evaluate DG spatial RHS (inverse_mass * residual)
    dU_dt = op.rhs(states)
    assert set(dU_dt.keys()) == set(states.keys())
    for elem_type in states:
        res = dU_dt[elem_type]
        assert res.shape == states[elem_type].shape
        assert jnp.all(jnp.isfinite(res))
        # For a uniform state with periodic / compatible boundary, interior flux divergence is 0
        print(f"P=3 {elem_type} residual max magnitude: {float(jnp.max(jnp.abs(res))):.4e}")


if __name__ == "__main__":
    print("Running test_p3_reference_elements...")
    test_p3_reference_elements()
    print("Running test_p3_exact_polynomial_integration...")
    test_p3_exact_polynomial_integration()
    print("Running test_p3_hybrid_spatial_operator...")
    test_p3_hybrid_spatial_operator()
    print("ALL P=3 HIGH-ORDER DG TESTS PASSED SUCCESSFULLY!")
