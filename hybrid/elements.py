"""Reference-element operators for type-blocked hybrid DG kernels."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ReferenceElement:
    element_type: str
    polynomial_degree: int
    nodes: jnp.ndarray
    volume_points: jnp.ndarray
    volume_weights: jnp.ndarray
    mass_matrix: jnp.ndarray
    inverse_mass_matrix: jnp.ndarray

    def basis(self, points):
        if self.polynomial_degree == 1:
            return p1_basis(self.element_type, points)
        if self.polynomial_degree == 2:
            return p2_basis(self.element_type, points)
        return p3_basis(self.element_type, points)

    def basis_gradients(self, points):
        if self.polynomial_degree == 1:
            return p1_basis_gradients(self.element_type, points)
        if self.polynomial_degree == 2:
            return p2_basis_gradients(self.element_type, points)
        return p3_basis_gradients(self.element_type, points)


def reference_nodes(element_type):
    if element_type == "tetra":
        return jnp.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )
    if element_type == "wedge":
        return jnp.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1],
            ],
            dtype=float,
        )
    if element_type == "hexahedron":
        return jnp.asarray(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def volume_quadrature(element_type, polynomial_degree=1):
    if polynomial_degree in {2, 3}:
        # Positive tensor-product Gauss rules under Duffy transformations.
        order = 5 if (polynomial_degree == 3 or element_type in {"tetra", "wedge"}) else 3
        raw, raw_weights = np.polynomial.legendre.leggauss(order)
        line = 0.5 * (raw + 1.0)
        weights_1d = 0.5 * raw_weights
        points = []
        weights = []
        if element_type == "tetra":
            for a, wa in zip(line, weights_1d):
                for b, wb in zip(line, weights_1d):
                    for c, wc in zip(line, weights_1d):
                        points.append((a, (1-a)*b, (1-a)*(1-b)*c))
                        weights.append(wa*wb*wc*(1-a)**2*(1-b))
        elif element_type == "wedge":
            for z, wz in zip(line, weights_1d):
                for a, wa in zip(line, weights_1d):
                    for b, wb in zip(line, weights_1d):
                        points.append((a, (1-a)*b, z))
                        weights.append(wa*wb*wz*(1-a))
        elif element_type == "hexahedron":
            for z, wz in zip(line, weights_1d):
                for y, wy in zip(line, weights_1d):
                    for x, wx in zip(line, weights_1d):
                        points.append((x, y, z))
                        weights.append(wx*wy*wz)
        else:
            raise ValueError(f"Unsupported element type {element_type!r}")
        return jnp.asarray(points), jnp.asarray(weights)

    if element_type == "tetra":
        a = 0.5854101966249685
        b = 0.1381966011250105
        points = jnp.asarray(
            [[b, b, b], [a, b, b], [b, a, b], [b, b, a]]
        )
        return points, jnp.full(4, 1.0 / 24.0)

    line_offset = 1.0 / np.sqrt(3.0)
    line_points = 0.5 * (1.0 + np.array([-line_offset, line_offset]))
    line_weights = np.full(2, 0.5)

    if element_type == "wedge":
        triangle_points = np.array(
            [[1.0 / 6.0, 1.0 / 6.0],
             [2.0 / 3.0, 1.0 / 6.0],
             [1.0 / 6.0, 2.0 / 3.0]]
        )
        triangle_weights = np.full(3, 1.0 / 6.0)
        points = []
        weights = []
        for z, wz in zip(line_points, line_weights):
            for (x, y), wt in zip(triangle_points, triangle_weights):
                points.append((x, y, z))
                weights.append(wt * wz)
        return jnp.asarray(points), jnp.asarray(weights)

    if element_type == "hexahedron":
        points = []
        weights = []
        for z, wz in zip(line_points, line_weights):
            for y, wy in zip(line_points, line_weights):
                for x, wx in zip(line_points, line_weights):
                    points.append((x, y, z))
                    weights.append(wx * wy * wz)
        return jnp.asarray(points), jnp.asarray(weights)

    raise ValueError(f"Unsupported element type {element_type!r}")


def p1_basis(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        return jnp.stack([1.0 - x - y - z, x, y, z], axis=-1)
    if element_type == "wedge":
        triangle = (1.0 - x - y, x, y)
        return jnp.stack(
            [
                triangle[0] * (1.0 - z),
                triangle[1] * (1.0 - z),
                triangle[2] * (1.0 - z),
                triangle[0] * z,
                triangle[1] * z,
                triangle[2] * z,
            ],
            axis=-1,
        )
    if element_type == "hexahedron":
        return jnp.stack(
            [
                (1-x)*(1-y)*(1-z), x*(1-y)*(1-z),
                x*y*(1-z), (1-x)*y*(1-z),
                (1-x)*(1-y)*z, x*(1-y)*z,
                x*y*z, (1-x)*y*z,
            ],
            axis=-1,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def p1_basis_gradients(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        constant = jnp.asarray(
            [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
        return jnp.broadcast_to(constant, points.shape[:-1] + constant.shape)
    if element_type == "wedge":
        triangle = (1.0 - x - y, x, y)
        bottom = 1.0 - z
        return jnp.stack(
            [
                jnp.stack([-bottom, -bottom, -triangle[0]], axis=-1),
                jnp.stack([ bottom, 0*bottom, -triangle[1]], axis=-1),
                jnp.stack([0*bottom,  bottom, -triangle[2]], axis=-1),
                jnp.stack([-z, -z, triangle[0]], axis=-1),
                jnp.stack([ z, 0*z, triangle[1]], axis=-1),
                jnp.stack([0*z,  z, triangle[2]], axis=-1),
            ],
            axis=-2,
        )
    if element_type == "hexahedron":
        values = (
            (-(1-y)*(1-z), -(1-x)*(1-z), -(1-x)*(1-y)),
            ((1-y)*(1-z), -x*(1-z), -x*(1-y)),
            (y*(1-z), x*(1-z), -x*y),
            (-y*(1-z), (1-x)*(1-z), -(1-x)*y),
            (-(1-y)*z, -(1-x)*z, (1-x)*(1-y)),
            ((1-y)*z, -x*z, x*(1-y)),
            (y*z, x*z, x*y),
            (-y*z, (1-x)*z, (1-x)*y),
        )
        return jnp.stack(
            [jnp.stack(gradient, axis=-1) for gradient in values], axis=-2
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def _quadratic_line_basis(value):
    # Nodes are ordered 0, 1, 1/2.
    return jnp.stack(
        [(1-value)*(1-2*value), value*(2*value-1), 4*value*(1-value)],
        axis=-1,
    )


def _quadratic_triangle_basis(x, y):
    l0, l1, l2 = 1.0 - x - y, x, y
    return jnp.stack(
        [
            l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1),
            4*l0*l1, 4*l1*l2, 4*l2*l0,
        ],
        axis=-1,
    )


def p2_reference_nodes(element_type):
    if element_type == "tetra":
        return jnp.asarray(
            [
                [0,0,0], [1,0,0], [0,1,0], [0,0,1],
                [.5,0,0], [0,.5,0], [0,0,.5],
                [.5,.5,0], [.5,0,.5], [0,.5,.5],
            ], dtype=float,
        )
    if element_type == "wedge":
        triangle = np.array(
            [[0,0], [1,0], [0,1], [.5,0], [.5,.5], [0,.5]]
        )
        return jnp.asarray(
            [[x,y,z] for z in (0.0, 1.0, 0.5) for x,y in triangle],
            dtype=float,
        )
    if element_type == "hexahedron":
        return jnp.asarray(
            [[x,y,z] for z in (0.0,1.0,0.5)
             for y in (0.0,1.0,0.5) for x in (0.0,1.0,0.5)],
            dtype=float,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def p2_basis(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        l0, l1, l2, l3 = 1-x-y-z, x, y, z
        return jnp.stack(
            [
                l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1), l3*(2*l3-1),
                4*l0*l1, 4*l0*l2, 4*l0*l3,
                4*l1*l2, 4*l1*l3, 4*l2*l3,
            ], axis=-1,
        )
    if element_type == "wedge":
        triangle = _quadratic_triangle_basis(x, y)
        line = _quadratic_line_basis(z)
        return jnp.concatenate(
            [triangle*line[..., index, None] for index in range(3)], axis=-1
        )
    if element_type == "hexahedron":
        bx = _quadratic_line_basis(x)
        by = _quadratic_line_basis(y)
        bz = _quadratic_line_basis(z)
        return jnp.stack(
            [bx[..., ix]*by[..., iy]*bz[..., iz]
             for iz in range(3) for iy in range(3) for ix in range(3)],
            axis=-1,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def _quadratic_line_derivative(value):
    return jnp.stack([4.0 * value - 3.0, 4.0 * value - 1.0, 4.0 - 8.0 * value], axis=-1)


def _quadratic_triangle_gradients(x, y):
    l0, l1, l2 = 1.0 - x - y, x, y
    d0 = jnp.stack([4.0 * l0 - 1.0, 0.0 * l0, 0.0 * l0, 4.0 * l1, 0.0 * l0, 4.0 * l2], axis=-1)
    d1 = jnp.stack([0.0 * l0, 4.0 * l1 - 1.0, 0.0 * l0, 4.0 * l0, 4.0 * l2, 0.0 * l0], axis=-1)
    d2 = jnp.stack([0.0 * l0, 0.0 * l0, 4.0 * l2 - 1.0, 0.0 * l0, 4.0 * l1, 4.0 * l0], axis=-1)
    return jnp.stack([-d0 + d1, -d0 + d2], axis=-1)


def p2_basis_gradients(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        l0, l1, l2, l3 = 1.0 - x - y - z, x, y, z
        d0 = jnp.stack([4.0*l0 - 1.0, 0.0*l0, 0.0*l0, 0.0*l0, 4.0*l1, 4.0*l2, 4.0*l3, 0.0*l0, 0.0*l0, 0.0*l0], axis=-1)
        d1 = jnp.stack([0.0*l0, 4.0*l1 - 1.0, 0.0*l0, 0.0*l0, 4.0*l0, 0.0*l0, 0.0*l0, 4.0*l2, 4.0*l3, 0.0*l0], axis=-1)
        d2 = jnp.stack([0.0*l0, 0.0*l0, 4.0*l2 - 1.0, 0.0*l0, 0.0*l0, 4.0*l0, 0.0*l0, 4.0*l1, 0.0*l0, 4.0*l3], axis=-1)
        d3 = jnp.stack([0.0*l0, 0.0*l0, 0.0*l0, 4.0*l3 - 1.0, 0.0*l0, 0.0*l0, 4.0*l0, 0.0*l0, 4.0*l1, 4.0*l2], axis=-1)
        return jnp.stack([-d0 + d1, -d0 + d2, -d0 + d3], axis=-1)

    if element_type == "wedge":
        tri_b = _quadratic_triangle_basis(x, y)
        tri_g = _quadratic_triangle_gradients(x, y)
        line_b = _quadratic_line_basis(z)
        line_d = _quadratic_line_derivative(z)
        gx = jnp.concatenate([tri_g[..., 0] * line_b[..., idx, None] for idx in range(3)], axis=-1)
        gy = jnp.concatenate([tri_g[..., 1] * line_b[..., idx, None] for idx in range(3)], axis=-1)
        gz = jnp.concatenate([tri_b * line_d[..., idx, None] for idx in range(3)], axis=-1)
        return jnp.stack([gx, gy, gz], axis=-1)

    if element_type == "hexahedron":
        bx = _quadratic_line_basis(x)
        by = _quadratic_line_basis(y)
        bz = _quadratic_line_basis(z)
        dbx = _quadratic_line_derivative(x)
        dby = _quadratic_line_derivative(y)
        dbz = _quadratic_line_derivative(z)
        gx = jnp.stack([dbx[..., ix] * by[..., iy] * bz[..., iz] for iz in range(3) for iy in range(3) for ix in range(3)], axis=-1)
        gy = jnp.stack([bx[..., ix] * dby[..., iy] * bz[..., iz] for iz in range(3) for iy in range(3) for ix in range(3)], axis=-1)
        gz = jnp.stack([bx[..., ix] * by[..., iy] * dbz[..., iz] for iz in range(3) for iy in range(3) for ix in range(3)], axis=-1)
        return jnp.stack([gx, gy, gz], axis=-1)

    raise ValueError(f"Unsupported element type {element_type!r}")


def _cubic_line_basis(value):
    # Nodes are ordered 0, 1, 1/3, 2/3.
    return jnp.stack(
        [
            -0.5 * (value - 1.0) * (3.0 * value - 1.0) * (3.0 * value - 2.0),
            0.5 * value * (3.0 * value - 1.0) * (3.0 * value - 2.0),
            4.5 * value * (1.0 - value) * (2.0 - 3.0 * value),
            4.5 * value * (1.0 - value) * (3.0 * value - 1.0),
        ],
        axis=-1,
    )


def _cubic_triangle_basis(x, y):
    l0, l1, l2 = 1.0 - x - y, x, y
    return jnp.stack(
        [
            # 3 Vertices
            0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0),
            0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0),
            0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0),
            # Edge 0-1
            4.5 * l0 * l1 * (3.0 * l0 - 1.0),
            4.5 * l0 * l1 * (3.0 * l1 - 1.0),
            # Edge 1-2
            4.5 * l1 * l2 * (3.0 * l1 - 1.0),
            4.5 * l1 * l2 * (3.0 * l2 - 1.0),
            # Edge 2-0
            4.5 * l0 * l2 * (3.0 * l2 - 1.0),
            4.5 * l0 * l2 * (3.0 * l0 - 1.0),
            # 1 Face Center
            27.0 * l0 * l1 * l2,
        ],
        axis=-1,
    )


def p3_reference_nodes(element_type):
    if element_type == "tetra":
        return jnp.asarray(
            [
                # 4 Vertices
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                # 12 Edge nodes
                [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0],
                [0.0, 1.0 / 3.0, 0.0], [0.0, 2.0 / 3.0, 0.0],
                [0.0, 0.0, 1.0 / 3.0], [0.0, 0.0, 2.0 / 3.0],
                [2.0 / 3.0, 1.0 / 3.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 1.0 / 3.0], [1.0 / 3.0, 0.0, 2.0 / 3.0],
                [0.0, 2.0 / 3.0, 1.0 / 3.0], [0.0, 1.0 / 3.0, 2.0 / 3.0],
                # 4 Face centers
                [1.0 / 3.0, 1.0 / 3.0, 0.0],
                [1.0 / 3.0, 0.0, 1.0 / 3.0],
                [0.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=float,
        )
    if element_type == "wedge":
        triangle = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                [1.0 / 3.0, 0.0], [2.0 / 3.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0],
                [0.0, 2.0 / 3.0], [0.0, 1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0],
            ]
        )
        return jnp.asarray(
            [[x, y, z] for z in (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0) for x, y in triangle],
            dtype=float,
        )
    if element_type == "hexahedron":
        line = (0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0)
        return jnp.asarray(
            [[x, y, z] for z in line for y in line for x in line],
            dtype=float,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def p3_basis(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        l0, l1, l2, l3 = 1.0 - x - y - z, x, y, z
        return jnp.stack(
            [
                # 4 Vertices
                0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0),
                0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0),
                0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0),
                0.5 * l3 * (3.0 * l3 - 1.0) * (3.0 * l3 - 2.0),
                # 12 Edge nodes
                4.5 * l0 * l1 * (3.0 * l0 - 1.0),
                4.5 * l0 * l1 * (3.0 * l1 - 1.0),
                4.5 * l0 * l2 * (3.0 * l0 - 1.0),
                4.5 * l0 * l2 * (3.0 * l2 - 1.0),
                4.5 * l0 * l3 * (3.0 * l0 - 1.0),
                4.5 * l0 * l3 * (3.0 * l3 - 1.0),
                4.5 * l1 * l2 * (3.0 * l1 - 1.0),
                4.5 * l1 * l2 * (3.0 * l2 - 1.0),
                4.5 * l1 * l3 * (3.0 * l1 - 1.0),
                4.5 * l1 * l3 * (3.0 * l3 - 1.0),
                4.5 * l2 * l3 * (3.0 * l2 - 1.0),
                4.5 * l2 * l3 * (3.0 * l3 - 1.0),
                # 4 Face centers
                27.0 * l0 * l1 * l2,
                27.0 * l0 * l1 * l3,
                27.0 * l0 * l2 * l3,
                27.0 * l1 * l2 * l3,
            ],
            axis=-1,
        )
    if element_type == "wedge":
        triangle = _cubic_triangle_basis(x, y)
        line = _cubic_line_basis(z)
        return jnp.concatenate(
            [triangle * line[..., index, None] for index in range(4)], axis=-1
        )
    if element_type == "hexahedron":
        bx = _cubic_line_basis(x)
        by = _cubic_line_basis(y)
        bz = _cubic_line_basis(z)
        return jnp.stack(
            [
                bx[..., ix] * by[..., iy] * bz[..., iz]
                for iz in range(4)
                for iy in range(4)
                for ix in range(4)
            ],
            axis=-1,
        )
    raise ValueError(f"Unsupported element type {element_type!r}")


def _cubic_line_derivative(value):
    return jnp.stack(
        [
            -0.5 * (27.0 * value**2 - 36.0 * value + 11.0),
            0.5 * (27.0 * value**2 - 18.0 * value + 2.0),
            4.5 * (9.0 * value**2 - 10.0 * value + 2.0),
            4.5 * (-9.0 * value**2 + 8.0 * value - 1.0),
        ],
        axis=-1,
    )


def _cubic_triangle_gradients(x, y):
    l0, l1, l2 = 1.0 - x - y, x, y
    d_dl0 = jnp.stack(
        [
            0.5 * (27.0 * l0**2 - 18.0 * l0 + 2.0),
            0.0 * l0,
            0.0 * l0,
            4.5 * l1 * (6.0 * l0 - 1.0),
            4.5 * l1 * (3.0 * l1 - 1.0),
            0.0 * l0,
            0.0 * l0,
            4.5 * l2 * (3.0 * l2 - 1.0),
            4.5 * l2 * (6.0 * l0 - 1.0),
            27.0 * l1 * l2,
        ],
        axis=-1,
    )
    d_dl1 = jnp.stack(
        [
            0.0 * l0,
            0.5 * (27.0 * l1**2 - 18.0 * l1 + 2.0),
            0.0 * l0,
            4.5 * l0 * (3.0 * l0 - 1.0),
            4.5 * l0 * (6.0 * l1 - 1.0),
            4.5 * l2 * (6.0 * l1 - 1.0),
            4.5 * l2 * (3.0 * l2 - 1.0),
            0.0 * l0,
            0.0 * l0,
            27.0 * l0 * l2,
        ],
        axis=-1,
    )
    d_dl2 = jnp.stack(
        [
            0.0 * l0,
            0.0 * l0,
            0.5 * (27.0 * l2**2 - 18.0 * l2 + 2.0),
            0.0 * l0,
            0.0 * l0,
            4.5 * l1 * (3.0 * l1 - 1.0),
            4.5 * l1 * (6.0 * l2 - 1.0),
            4.5 * l0 * (6.0 * l2 - 1.0),
            4.5 * l0 * (3.0 * l0 - 1.0),
            27.0 * l0 * l1,
        ],
        axis=-1,
    )
    dx = -d_dl0 + d_dl1
    dy = -d_dl0 + d_dl2
    return jnp.stack([dx, dy], axis=-1)


def p3_basis_gradients(element_type, points):
    points = jnp.asarray(points)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    if element_type == "tetra":
        l0, l1, l2, l3 = 1.0 - x - y - z, x, y, z
        d_dl0 = jnp.stack(
            [
                0.5 * (27.0 * l0**2 - 18.0 * l0 + 2.0),
                0.0 * l0, 0.0 * l0, 0.0 * l0,
                4.5 * l1 * (6.0 * l0 - 1.0),
                4.5 * l1 * (3.0 * l1 - 1.0),
                4.5 * l2 * (6.0 * l0 - 1.0),
                4.5 * l2 * (3.0 * l2 - 1.0),
                4.5 * l3 * (6.0 * l0 - 1.0),
                4.5 * l3 * (3.0 * l3 - 1.0),
                0.0 * l0, 0.0 * l0, 0.0 * l0, 0.0 * l0, 0.0 * l0, 0.0 * l0,
                27.0 * l1 * l2,
                27.0 * l1 * l3,
                27.0 * l2 * l3,
                0.0 * l0,
            ],
            axis=-1,
        )
        d_dl1 = jnp.stack(
            [
                0.0 * l0,
                0.5 * (27.0 * l1**2 - 18.0 * l1 + 2.0),
                0.0 * l0, 0.0 * l0,
                4.5 * l0 * (3.0 * l0 - 1.0),
                4.5 * l0 * (6.0 * l1 - 1.0),
                0.0 * l0, 0.0 * l0, 0.0 * l0, 0.0 * l0,
                4.5 * l2 * (6.0 * l1 - 1.0),
                4.5 * l2 * (3.0 * l2 - 1.0),
                4.5 * l3 * (6.0 * l1 - 1.0),
                4.5 * l3 * (3.0 * l3 - 1.0),
                0.0 * l0, 0.0 * l0,
                27.0 * l0 * l2,
                27.0 * l0 * l3,
                0.0 * l0,
                27.0 * l2 * l3,
            ],
            axis=-1,
        )
        d_dl2 = jnp.stack(
            [
                0.0 * l0, 0.0 * l0,
                0.5 * (27.0 * l2**2 - 18.0 * l2 + 2.0),
                0.0 * l0, 0.0 * l0, 0.0 * l0,
                4.5 * l0 * (3.0 * l0 - 1.0),
                4.5 * l0 * (6.0 * l2 - 1.0),
                0.0 * l0, 0.0 * l0,
                4.5 * l1 * (3.0 * l1 - 1.0),
                4.5 * l1 * (6.0 * l2 - 1.0),
                0.0 * l0, 0.0 * l0,
                4.5 * l3 * (6.0 * l2 - 1.0),
                4.5 * l3 * (3.0 * l3 - 1.0),
                27.0 * l0 * l1,
                0.0 * l0,
                27.0 * l0 * l3,
                27.0 * l1 * l3,
            ],
            axis=-1,
        )
        d_dl3 = jnp.stack(
            [
                0.0 * l0, 0.0 * l0, 0.0 * l0,
                0.5 * (27.0 * l3**2 - 18.0 * l3 + 2.0),
                0.0 * l0, 0.0 * l0, 0.0 * l0, 0.0 * l0,
                4.5 * l0 * (3.0 * l0 - 1.0),
                4.5 * l0 * (6.0 * l3 - 1.0),
                0.0 * l0, 0.0 * l0,
                4.5 * l1 * (3.0 * l1 - 1.0),
                4.5 * l1 * (6.0 * l3 - 1.0),
                4.5 * l2 * (3.0 * l2 - 1.0),
                4.5 * l2 * (6.0 * l3 - 1.0),
                0.0 * l0,
                27.0 * l0 * l1,
                27.0 * l0 * l2,
                27.0 * l1 * l2,
            ],
            axis=-1,
        )
        dx = -d_dl0 + d_dl1
        dy = -d_dl0 + d_dl2
        dz = -d_dl0 + d_dl3
        return jnp.stack([dx, dy, dz], axis=-1)

    if element_type == "wedge":
        tri_b = _cubic_triangle_basis(x, y)
        tri_g = _cubic_triangle_gradients(x, y)
        line_b = _cubic_line_basis(z)
        line_d = _cubic_line_derivative(z)

        # Concatenate 4 z-levels of triangle basis functions
        gx = jnp.concatenate(
            [tri_g[..., 0] * line_b[..., idx, None] for idx in range(4)], axis=-1
        )
        gy = jnp.concatenate(
            [tri_g[..., 1] * line_b[..., idx, None] for idx in range(4)], axis=-1
        )
        gz = jnp.concatenate(
            [tri_b * line_d[..., idx, None] for idx in range(4)], axis=-1
        )
        return jnp.stack([gx, gy, gz], axis=-1)

    if element_type == "hexahedron":
        bx = _cubic_line_basis(x)
        by = _cubic_line_basis(y)
        bz = _cubic_line_basis(z)
        dbx = _cubic_line_derivative(x)
        dby = _cubic_line_derivative(y)
        dbz = _cubic_line_derivative(z)

        gx = jnp.stack(
            [
                dbx[..., ix] * by[..., iy] * bz[..., iz]
                for iz in range(4)
                for iy in range(4)
                for ix in range(4)
            ],
            axis=-1,
        )
        gy = jnp.stack(
            [
                bx[..., ix] * dby[..., iy] * bz[..., iz]
                for iz in range(4)
                for iy in range(4)
                for ix in range(4)
            ],
            axis=-1,
        )
        gz = jnp.stack(
            [
                bx[..., ix] * by[..., iy] * dbz[..., iz]
                for iz in range(4)
                for iy in range(4)
                for ix in range(4)
            ],
            axis=-1,
        )
        return jnp.stack([gx, gy, gz], axis=-1)

    raise ValueError(f"Unsupported element type {element_type!r}")


def make_reference_element(element_type, polynomial_degree=1):
    if polynomial_degree not in {1, 2, 3}:
        raise ValueError("Only P1, P2, and P3 are currently implemented")
    if polynomial_degree == 1:
        nodes = reference_nodes(element_type)
        basis_fn = p1_basis
    elif polynomial_degree == 2:
        nodes = p2_reference_nodes(element_type)
        basis_fn = p2_basis
    else:
        nodes = p3_reference_nodes(element_type)
        basis_fn = p3_basis

    points, weights = volume_quadrature(element_type, polynomial_degree)
    basis = basis_fn(element_type, points)
    mass = jnp.einsum(
        "q,qi,qj->ij",
        weights,
        basis,
        basis,
        precision=jax.lax.Precision.HIGHEST,
    )
    return ReferenceElement(
        element_type=element_type,
        polynomial_degree=polynomial_degree,
        nodes=nodes,
        volume_points=points,
        volume_weights=weights,
        mass_matrix=mass,
        inverse_mass_matrix=jnp.linalg.inv(mass),
    )

