"""Conservative reference-subcell operators for a-posteriori DG/FV limiting.

The objects in this module contain no time integration policy.  They provide
the geometric and algebraic pieces required by a MOOD stage: exact-enough
metric-weighted DG-to-FV projection, conservative reconstruction, and a
conforming subdivision of P2 tetrahedron and wedge reference elements.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .elements import p1_basis, volume_quadrature
from .geometry import LOCAL_FACES


@dataclass(frozen=True)
class ReferenceSubcells:
    element_type: str
    division_count: int
    vertices: np.ndarray
    cells: np.ndarray
    internal_owner: np.ndarray
    internal_neighbor: np.ndarray
    internal_faces: tuple[np.ndarray, ...]
    boundary_owner: np.ndarray
    boundary_parent_face: np.ndarray
    boundary_faces: tuple[np.ndarray, ...]

    @property
    def count(self):
        return len(self.cells)


@dataclass(frozen=True)
class SubcellProjection:
    projection: jnp.ndarray
    reconstruction: jnp.ndarray
    volumes: jnp.ndarray
    reference_centers: jnp.ndarray
    physical_centers: jnp.ndarray
    internal_normals: jnp.ndarray
    internal_areas: jnp.ndarray
    boundary_normals: jnp.ndarray
    boundary_areas: jnp.ndarray
    boundary_reference_centers: jnp.ndarray
    boundary_physical_centers: jnp.ndarray


def _triangle_lattice(division_count):
    n = division_count
    vertices = np.asarray(
        [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)],
        dtype=np.float64,
    )
    lookup = {
        tuple(np.rint(point * n).astype(int)): index
        for index, point in enumerate(vertices)
    }
    triangles = []
    for i in range(n):
        for j in range(n - i):
            triangles.append([lookup[(i, j)], lookup[(i + 1, j)], lookup[(i, j + 1)]])
            if i + j <= n - 2:
                triangles.append([
                    lookup[(i + 1, j)],
                    lookup[(i + 1, j + 1)],
                    lookup[(i, j + 1)],
                ])
    return vertices, np.asarray(triangles, dtype=np.int32)


def _tetra_subdivision(division_count):
    if division_count != 4:
        raise ValueError("The conforming P2 tetra subdivision uses 4 divisions")
    vertices = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    cells = [(0, 1, 2, 3)]
    # Two uniform red refinements give 4 subdivisions per edge and 64
    # conforming tetrahedra.  Unlike Delaunay on a co-spherical lattice, this
    # has identical face diagonals on both sides of every internal face.
    for _ in range(2):
        lookup = {tuple(point): i for i, point in enumerate(vertices)}

        def midpoint(first, second):
            point = 0.5 * (vertices[first] + vertices[second])
            key = tuple(point)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(point)
            return lookup[key]

        refined = []
        for a, b, c, d in cells:
            ab, ac, ad = midpoint(a, b), midpoint(a, c), midpoint(a, d)
            bc, bd, cd = midpoint(b, c), midpoint(b, d), midpoint(c, d)
            refined.extend([
                (a, ab, ac, ad), (ab, b, bc, bd),
                (ac, bc, c, cd), (ad, bd, cd, d),
                (ab, ac, ad, bd), (ab, bc, ac, bd),
                (ac, ad, bd, cd), (ac, bd, bc, cd),
            ])
        cells = refined
    return np.asarray(vertices), np.asarray(cells, dtype=np.int32)


def _wedge_subdivision(division_count):
    triangle_vertices, triangles = _triangle_lattice(division_count)
    n = division_count
    vertices = np.asarray(
        [
            (x, y, layer / n)
            for layer in range(n + 1)
            for x, y in triangle_vertices
        ],
        dtype=np.float64,
    )
    layer_size = len(triangle_vertices)
    cells = []
    for layer in range(n):
        lower = layer * layer_size
        upper = (layer + 1) * layer_size
        for a, b, c in triangles:
            cells.append([lower + a, lower + b, lower + c,
                          upper + a, upper + b, upper + c])
    return vertices, np.asarray(cells, dtype=np.int32)


def _parent_face_id(element_type, coordinates, tolerance=1.0e-10):
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    tests = (
        (z,) if element_type == "tetra" else (z, z - 1.0)
    )
    if element_type == "tetra":
        tests = (z, y, x + y + z - 1.0, x)
    elif element_type == "wedge":
        tests = (z, z - 1.0, y, x + y - 1.0, x)
    else:
        raise ValueError(element_type)
    for face, value in enumerate(tests):
        if np.all(np.abs(value) <= tolerance):
            return face
    return -1


def _topology(element_type, vertices, cells):
    local_faces = LOCAL_FACES[element_type]
    pending = {}
    internal_owner, internal_neighbor, internal_faces = [], [], []
    boundary_owner, boundary_parent, boundary_faces = [], [], []
    for cell, connectivity in enumerate(cells):
        for local in local_faces:
            face = np.asarray(connectivity[list(local)], dtype=np.int32)
            key = tuple(sorted(int(index) for index in face))
            if key in pending:
                other_cell, other_face = pending.pop(key)
                internal_owner.append(other_cell)
                internal_neighbor.append(cell)
                internal_faces.append(other_face)
            else:
                pending[key] = (cell, face)
    for cell, face in pending.values():
        parent_face = _parent_face_id(element_type, vertices[face])
        if parent_face < 0:
            raise ValueError("Subcell boundary face is not on its parent boundary")
        boundary_owner.append(cell)
        boundary_parent.append(parent_face)
        boundary_faces.append(face)
    return (
        np.asarray(internal_owner, dtype=np.int32),
        np.asarray(internal_neighbor, dtype=np.int32),
        tuple(internal_faces),
        np.asarray(boundary_owner, dtype=np.int32),
        np.asarray(boundary_parent, dtype=np.int32),
        tuple(boundary_faces),
    )


def make_reference_subcells(element_type, division_count=4):
    if division_count != 4:
        raise ValueError("P2 subcell coupling currently requires division_count=4")
    if element_type == "tetra":
        vertices, cells = _tetra_subdivision(division_count)
    elif element_type == "wedge":
        vertices, cells = _wedge_subdivision(division_count)
    else:
        raise ValueError(f"Unsupported subcell parent {element_type!r}")
    return ReferenceSubcells(
        element_type,
        division_count,
        vertices,
        cells,
        *_topology(element_type, vertices, cells),
    )


def _mapped_quadrature(reference_subcells):
    element_type = reference_subcells.element_type
    points, weights = volume_quadrature(element_type, polynomial_degree=2)
    points = np.asarray(points)
    weights = np.asarray(weights)
    cell_vertices = reference_subcells.vertices[reference_subcells.cells]
    basis = np.asarray(p1_basis(element_type, points))
    mapped = np.einsum("qi,sir->sqr", basis, cell_vertices)
    if element_type == "tetra":
        matrices = np.stack(
            [
                cell_vertices[:, 1] - cell_vertices[:, 0],
                cell_vertices[:, 2] - cell_vertices[:, 0],
                cell_vertices[:, 3] - cell_vertices[:, 0],
            ],
            axis=-1,
        )
    else:
        matrices = np.stack(
            [
                cell_vertices[:, 1] - cell_vertices[:, 0],
                cell_vertices[:, 2] - cell_vertices[:, 0],
                cell_vertices[:, 3] - cell_vertices[:, 0],
            ],
            axis=-1,
        )
    local_scale = np.abs(np.linalg.det(matrices))
    return jnp.asarray(mapped), jnp.asarray(weights[None, :] * local_scale[:, None])


def build_subcell_projection(operator, block_id, reference_subcells, chunk_size=256):
    """Build physical, metric-weighted projection and its P2 left inverse."""
    block = operator.mesh.element_blocks[block_id]
    element_type = block.element_type
    reference = operator.geometry.reference_elements[element_type]
    geometry_reference = operator.geometry.geometry_reference_elements[element_type]
    geometry_connectivity = (
        block.geometry_connectivity
        if geometry_reference.polynomial_degree == 2
        else block.connectivity
    )
    coordinates = jnp.asarray(operator.mesh.points[geometry_connectivity])
    reference_vertices = jnp.asarray(reference_subcells.vertices)
    geometry_vertex_basis = geometry_reference.basis(reference_vertices)
    physical_vertices = jnp.einsum(
        "vi,eid->evd", geometry_vertex_basis, coordinates,
        precision=jax.lax.Precision.HIGHEST,
    )
    physical_cells = physical_vertices[:, jnp.asarray(reference_subcells.cells)]
    physical_centers = jnp.mean(physical_cells, axis=2)
    points, reference_weights = _mapped_quadrature(reference_subcells)
    flat_points = points.reshape((-1, 3))
    solution_basis = reference.basis(flat_points).reshape(
        (reference_subcells.count, points.shape[1], -1)
    )
    geometry_gradients = geometry_reference.basis_gradients(flat_points).reshape(
        (reference_subcells.count, points.shape[1], -1, 3)
    )
    projections = []
    volumes = []
    for start in range(0, len(block.connectivity), chunk_size):
        xyz = coordinates[start:start + chunk_size]
        jacobian = jnp.einsum(
            "eir,sqij->esqrj", xyz, geometry_gradients,
            precision=jax.lax.Precision.HIGHEST,
        )
        determinant = jnp.abs(jnp.linalg.det(jacobian))
        physical_weights = determinant * reference_weights[None, :, :]
        subcell_volume = jnp.sum(physical_weights, axis=-1)
        projection = jnp.einsum(
            "esq,sqi->esi", physical_weights, solution_basis,
            precision=jax.lax.Precision.HIGHEST,
        ) / subcell_volume[..., None]
        projections.append(projection)
        volumes.append(subcell_volume)
    projection = jnp.concatenate(projections, axis=0)
    volumes = jnp.concatenate(volumes, axis=0)
    # Volume-weighted least squares is a left inverse on the P2 polynomial
    # space.  It also gives physically smaller subcells appropriate influence.
    gram = jnp.einsum(
        "esi,es,esj->eij", projection, volumes, projection,
        precision=jax.lax.Precision.HIGHEST,
    )
    reconstruction = jnp.einsum(
        "eij,esj,es->eis",
        jnp.linalg.inv(gram), projection, volumes,
        precision=jax.lax.Precision.HIGHEST,
    )

    def face_metrics(face_vertices, owner_cells, neighbor_cells=None):
        metrics = []
        centers = []
        for face in face_vertices:
            xyz = physical_vertices[:, jnp.asarray(face)]
            center = jnp.mean(xyz, axis=1)
            area_vector = 0.5 * jnp.cross(
                xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
            )
            if len(face) == 4:
                area_vector = area_vector + 0.5 * jnp.cross(
                    xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]
                )
            metrics.append(area_vector)
            centers.append(center)
        area_vectors = jnp.stack(metrics, axis=1)
        centers = jnp.stack(centers, axis=1)
        owner_center = physical_centers[:, jnp.asarray(owner_cells)]
        if neighbor_cells is None:
            direction = centers - owner_center
        else:
            direction = (
                physical_centers[:, jnp.asarray(neighbor_cells)] - owner_center
            )
        orientation = jnp.where(
            jnp.sum(area_vectors * direction, axis=-1) >= 0.0, 1.0, -1.0
        )
        area_vectors = area_vectors * orientation[..., None]
        areas = jnp.linalg.norm(area_vectors, axis=-1)
        return area_vectors / areas[..., None], areas, centers

    internal_normals, internal_areas, _ = face_metrics(
        reference_subcells.internal_faces,
        reference_subcells.internal_owner,
        reference_subcells.internal_neighbor,
    )
    boundary_normals, boundary_areas, boundary_physical_centers = face_metrics(
        reference_subcells.boundary_faces,
        reference_subcells.boundary_owner,
    )
    boundary_reference_centers = jnp.asarray(
        np.asarray([
            np.mean(reference_subcells.vertices[face], axis=0)
            for face in reference_subcells.boundary_faces
        ])
    )
    return SubcellProjection(
        projection=projection,
        reconstruction=reconstruction,
        volumes=volumes,
        reference_centers=jnp.asarray(
            np.mean(reference_subcells.vertices[reference_subcells.cells], axis=1)
        ),
        physical_centers=physical_centers,
        internal_normals=internal_normals,
        internal_areas=internal_areas,
        boundary_normals=boundary_normals,
        boundary_areas=boundary_areas,
        boundary_reference_centers=boundary_reference_centers,
        boundary_physical_centers=boundary_physical_centers,
    )


def dg_to_subcells(U, projection):
    return jnp.einsum(
        "esi,eiv->esv", projection.projection, U,
        precision=jax.lax.Precision.HIGHEST,
    )


def subcells_to_dg(subcell_averages, projection):
    return jnp.einsum(
        "eis,esv->eiv", projection.reconstruction, subcell_averages,
        precision=jax.lax.Precision.HIGHEST,
    )


class HybridSubcellFV:
    """Conservative MOOD recomputation operator for P2 tetra/wedge blocks."""

    def __init__(self, operator, division_count=4, reconstruction="weno"):
        self.operator = operator
        self.solver = operator.solver
        self.mesh = operator.mesh
        self.division_count = division_count
        self.reconstruction = reconstruction
        self.references = {}
        self.projections = {}
        for block_id, block in enumerate(self.mesh.element_blocks):
            if block.element_type not in {"tetra", "wedge"}:
                raise ValueError(
                    "Subcell FV currently supports tetrahedron and wedge blocks"
                )
            reference = make_reference_subcells(
                block.element_type, division_count
            )
            self.references[block.element_type] = reference
            self.projections[block.element_type] = build_subcell_projection(
                operator, block_id, reference
            )
        self.face_groups = self._build_face_groups()

    def _boundary_indices(self, element_type, local_face):
        reference = self.references[element_type]
        return np.flatnonzero(
            reference.boundary_parent_face == local_face
        ).astype(np.int32)

    def _build_face_groups(self):
        groups = []
        physical_centers = {
            name: np.asarray(projection.boundary_physical_centers)
            for name, projection in self.projections.items()
        }
        for face_type, owner_block_id, neighbor_block_id, indices_jax in self.operator.face_groups:
            indices = np.asarray(indices_jax)
            faces = self.mesh.face_blocks[face_type]
            owner_block = self.mesh.element_blocks[owner_block_id]
            owner_projection = self.projections[owner_block.element_type]
            owner_subcells = []
            neighbor_subcells = []
            for face_index in indices:
                owner_local = int(faces.owner_local[face_index])
                owner_face = int(faces.owner_face[face_index])
                owner_ids = self._boundary_indices(
                    owner_block.element_type, owner_face
                )
                owner_subcells.append(owner_ids)
                if neighbor_block_id >= 0:
                    neighbor_block = self.mesh.element_blocks[neighbor_block_id]
                    neighbor_local = int(faces.neighbor_local[face_index])
                    neighbor_face = int(faces.neighbor_face[face_index])
                    neighbor_ids = self._boundary_indices(
                        neighbor_block.element_type, neighbor_face
                    )
                    owner_xyz = physical_centers[owner_block.element_type][
                        owner_local, owner_ids
                    ]
                    neighbor_xyz = physical_centers[neighbor_block.element_type][
                        neighbor_local, neighbor_ids
                    ]
                    distances = np.linalg.norm(
                        owner_xyz[:, None, :] - neighbor_xyz[None, :, :], axis=-1
                    )
                    permutation = np.argmin(distances, axis=1)
                    scale = max(
                        float(np.max(np.linalg.norm(owner_xyz, axis=1))), 1.0
                    )
                    if (
                        len(np.unique(permutation)) != len(owner_ids)
                        or np.max(distances[np.arange(len(owner_ids)), permutation])
                        > 2.0e-6 * scale
                    ):
                        raise ValueError(
                            f"Nonconforming subfaces at {face_type} face {face_index}"
                        )
                    neighbor_subcells.append(neighbor_ids[permutation])
            groups.append({
                "face_type": face_type,
                "owner_block": owner_block_id,
                "neighbor_block": neighbor_block_id,
                "indices": jnp.asarray(indices),
                "owner_subcells": jnp.asarray(np.asarray(owner_subcells)),
                "neighbor_subcells": (
                    None if neighbor_block_id < 0
                    else jnp.asarray(np.asarray(neighbor_subcells))
                ),
            })
        return groups

    def project(self, states):
        return {
            name: dg_to_subcells(U, self.projections[name])
            for name, U in states.items()
        }

    def _reconstruct_internal_states(self, name, U):
        reference = self.references[name]
        owner = jnp.asarray(reference.internal_owner)
        neighbor = jnp.asarray(reference.internal_neighbor)

        if self.reconstruction == "constant":
            return U[:, owner], U[:, neighbor]

        projection = self.projections[name]
        normals = projection.internal_normals
        areas = projection.internal_areas
        volumes = projection.volumes
        centers = projection.physical_centers

        num_cells = reference.count
        # Face difference vector across internal faces
        delta_U = U[:, neighbor] - U[:, owner]

        # 1. Green-Gauss gradient on subcells
        flux_geom = 0.5 * normals * areas[:, :, None]
        face_contrib = delta_U[:, :, None, :] * flux_geom[:, :, :, None]

        grad_U = jnp.zeros((U.shape[0], num_cells, 3, U.shape[-1]), dtype=U.dtype)
        grad_U = grad_U.at[:, owner].add(face_contrib)
        grad_U = grad_U.at[:, neighbor].add(face_contrib)
        grad_U = grad_U / jnp.maximum(volumes[:, :, None, None], 1.0e-14)

        # 2. Directional derivatives along centerline d_center = x_N - x_O
        d_center = centers[:, neighbor] - centers[:, owner]
        du_owner = jnp.sum(grad_U[:, owner] * d_center[:, :, :, None], axis=2)
        du_neighbor = jnp.sum(grad_U[:, neighbor] * (-d_center[:, :, :, None]), axis=2)

        if self.reconstruction == "tvd":
            # 3. Slope limiters (smooth Venkatakrishnan TVD)
            eps = 1.0e-12
            r_owner = delta_U / (du_owner + jnp.sign(du_owner) * eps)
            r_neighbor = (-delta_U) / (du_neighbor + jnp.sign(du_neighbor) * eps)

            phi_owner = jnp.clip(
                jnp.where(
                    r_owner > 0.0,
                    (r_owner**2 + 2.0 * r_owner) / (r_owner**2 + r_owner + 2.0),
                    0.0,
                ),
                0.0,
                1.0,
            )
            phi_neighbor = jnp.clip(
                jnp.where(
                    r_neighbor > 0.0,
                    (r_neighbor**2 + 2.0 * r_neighbor) / (r_neighbor**2 + r_neighbor + 2.0),
                    0.0,
                ),
                0.0,
                1.0,
            )

            U_left = U[:, owner] + 0.5 * phi_owner * du_owner
            U_right = U[:, neighbor] + 0.5 * phi_neighbor * du_neighbor

        elif self.reconstruction == "weno":
            # 3. Third-order Central WENO (CWENO-Z)
            # Candidate 0: High-order central cubic Hermite interpolant at face midpoint
            s_owner = du_owner
            s_neighbor = -du_neighbor
            P_high = 0.5 * (U[:, owner] + U[:, neighbor]) + 0.125 * (s_owner - s_neighbor)
            P_owner = U[:, owner] + 0.5 * s_owner
            P_neighbor = U[:, neighbor] - 0.5 * s_neighbor
            P_const_owner = U[:, owner]
            P_const_neighbor = U[:, neighbor]

            # Smoothness indicators
            eps_s = 1.0e-12
            s_secant = delta_U
            beta_high = (s_neighbor - s_owner) ** 2 + 0.1 * s_secant ** 2
            beta_owner = (s_secant - s_owner) ** 2 + 0.1 * s_secant ** 2
            beta_neighbor = (s_secant - s_neighbor) ** 2 + 0.1 * s_secant ** 2
            beta_const = s_secant ** 2

            # WENO-Z global oscillation indicator tau
            tau = jnp.abs(beta_const - beta_high)

            # Linear optimal weights: 70% central high-order, 28% directional, 2% constant
            d0, d1, d2 = 0.70, 0.28, 0.02

            # Left state weights (owner side)
            a0_L = d0 * (1.0 + (tau / (beta_high + eps_s)) ** 2)
            a1_L = d1 * (1.0 + (tau / (beta_owner + eps_s)) ** 2)
            a2_L = d2 * (1.0 + (tau / (beta_const + eps_s)) ** 2)
            sum_a_L = a0_L + a1_L + a2_L
            w0_L, w1_L, w2_L = a0_L / sum_a_L, a1_L / sum_a_L, a2_L / sum_a_L
            U_left = w0_L * P_high + w1_L * P_owner + w2_L * P_const_owner

            # Right state weights (neighbor side)
            a0_R = d0 * (1.0 + (tau / (beta_high + eps_s)) ** 2)
            a1_R = d1 * (1.0 + (tau / (beta_neighbor + eps_s)) ** 2)
            a2_R = d2 * (1.0 + (tau / (beta_const + eps_s)) ** 2)
            sum_a_R = a0_R + a1_R + a2_R
            w0_R, w1_R, w2_R = a0_R / sum_a_R, a1_R / sum_a_R, a2_R / sum_a_R
            U_right = w0_R * P_high + w1_R * P_neighbor + w2_R * P_const_neighbor

        else:
            raise ValueError(f"Unknown subcell reconstruction mode: {self.reconstruction!r}")

        # Admissibility preservation (positive density and internal energy)
        min_rho = 1.0e-14
        U_left = U_left.at[..., 0].set(jnp.maximum(U_left[..., 0], min_rho))
        U_right = U_right.at[..., 0].set(jnp.maximum(U_right[..., 0], min_rho))

        e_kin_L = 0.5 * jnp.sum(U_left[..., 1:4] ** 2, axis=-1) / U_left[..., 0]
        U_left = U_left.at[..., 4].set(
            jnp.maximum(U_left[..., 4], e_kin_L + 1.0e-12)
        )
        e_kin_R = 0.5 * jnp.sum(U_right[..., 1:4] ** 2, axis=-1) / U_right[..., 0]
        U_right = U_right.at[..., 4].set(
            jnp.maximum(U_right[..., 4], e_kin_R + 1.0e-12)
        )

        return U_left, U_right

    def _internal_residual(self, subcell_states):
        residual = {}
        for block in self.mesh.element_blocks:
            name = block.element_type
            reference = self.references[name]
            projection = self.projections[name]
            U = subcell_states[name]
            owner = jnp.asarray(reference.internal_owner)
            neighbor = jnp.asarray(reference.internal_neighbor)
            normals = projection.internal_normals
            areas = projection.internal_areas

            U_left, U_right = self._reconstruct_internal_states(name, U)

            flux = self.solver.numerical_flux(
                U_left, U_right, normals
            ) * areas[..., None]
            block_residual = jnp.zeros_like(U)
            block_residual = block_residual.at[:, owner, :].add(-flux)
            block_residual = block_residual.at[:, neighbor, :].add(flux)
            residual[name] = block_residual
        return residual

    def _dg_group_flux(self, states, group, boundary_state):
        face_type = group["face_type"]
        owner_block_id = group["owner_block"]
        neighbor_block_id = group["neighbor_block"]
        indices = group["indices"]
        faces = self.mesh.face_blocks[face_type]
        metrics = self.operator.geometry.face_metrics[face_type]
        owner_block = self.mesh.element_blocks[owner_block_id]
        owner_cells = jnp.asarray(faces.owner_local)[indices]
        owner_basis = self.operator.geometry.reference_elements[
            owner_block.element_type
        ].basis(metrics.owner_reference_points[indices])
        U_left = jnp.einsum(
            "fqi,fiv->fqv", owner_basis,
            states[owner_block.element_type][owner_cells],
            precision=jax.lax.Precision.HIGHEST,
        )
        normals = metrics.normals[indices]
        if neighbor_block_id >= 0:
            neighbor_block = self.mesh.element_blocks[neighbor_block_id]
            neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
            neighbor_basis = self.operator.geometry.reference_elements[
                neighbor_block.element_type
            ].basis(metrics.neighbor_reference_points[indices])
            U_right = jnp.einsum(
                "fqi,fiv->fqv", neighbor_basis,
                states[neighbor_block.element_type][neighbor_cells],
                precision=jax.lax.Precision.HIGHEST,
            )
        else:
            U_right = (
                U_left if boundary_state is None
                else boundary_state(
                    U_left, normals,
                    jnp.asarray(faces.boundary_tags)[indices],
                )
            )
        return jnp.sum(
            self.solver.numerical_flux(U_left, U_right, normals)
            * metrics.integration_weights[indices, :, None],
            axis=1,
        )

    def _external_fluxes(
        self, states, subcell_states, troubled, residual, boundary_state
    ):
        integrated_fluxes = []
        original_fluxes = []
        for group in self.face_groups:
            face_type = group["face_type"]
            indices = group["indices"]
            faces = self.mesh.face_blocks[face_type]
            owner_block_id = group["owner_block"]
            neighbor_block_id = group["neighbor_block"]
            owner_block = self.mesh.element_blocks[owner_block_id]
            owner_name = owner_block.element_type
            owner_cells = jnp.asarray(faces.owner_local)[indices]
            owner_subids = group["owner_subcells"]
            owner_projection = self.projections[owner_name]
            owner_ref_points = owner_projection.boundary_reference_centers[
                owner_subids
            ]
            owner_basis = self.operator.geometry.reference_elements[
                owner_name
            ].basis(owner_ref_points)
            owner_dg = jnp.einsum(
                "fki,fiv->fkv", owner_basis, states[owner_name][owner_cells],
                precision=jax.lax.Precision.HIGHEST,
            )
            owner_fv = subcell_states[owner_name][
                owner_cells[:, None], owner_subids
            ]
            owner_flag = troubled[owner_name][owner_cells]
            U_left = jnp.where(owner_flag[:, None, None], owner_fv, owner_dg)
            normals = owner_projection.boundary_normals[
                owner_cells[:, None], owner_subids
            ]
            areas = owner_projection.boundary_areas[
                owner_cells[:, None], owner_subids
            ]

            if neighbor_block_id >= 0:
                neighbor_block = self.mesh.element_blocks[neighbor_block_id]
                neighbor_name = neighbor_block.element_type
                neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
                neighbor_subids = group["neighbor_subcells"]
                neighbor_projection = self.projections[neighbor_name]
                neighbor_ref_points = (
                    neighbor_projection.boundary_reference_centers[
                        neighbor_subids
                    ]
                )
                neighbor_basis = self.operator.geometry.reference_elements[
                    neighbor_name
                ].basis(neighbor_ref_points)
                neighbor_dg = jnp.einsum(
                    "fki,fiv->fkv", neighbor_basis,
                    states[neighbor_name][neighbor_cells],
                    precision=jax.lax.Precision.HIGHEST,
                )
                neighbor_fv = subcell_states[neighbor_name][
                    neighbor_cells[:, None], neighbor_subids
                ]
                neighbor_flag = troubled[neighbor_name][neighbor_cells]
                U_right = jnp.where(
                    neighbor_flag[:, None, None], neighbor_fv, neighbor_dg
                )
            else:
                neighbor_name = None
                neighbor_cells = None
                neighbor_subids = None
                neighbor_flag = jnp.zeros_like(owner_flag)
                tags = jnp.asarray(faces.boundary_tags)[indices]
                U_right = (
                    U_left if boundary_state is None
                    else boundary_state(U_left, normals, tags)
                )

            flux = self.solver.numerical_flux(
                U_left, U_right, normals
            ) * areas[..., None]
            residual[owner_name] = residual[owner_name].at[
                owner_cells[:, None], owner_subids
            ].add(-flux)
            if neighbor_block_id >= 0:
                residual[neighbor_name] = residual[neighbor_name].at[
                    neighbor_cells[:, None], neighbor_subids
                ].add(flux)
            integrated_fluxes.append(jnp.sum(flux, axis=1))
            original_fluxes.append(
                self._dg_group_flux(states, group, boundary_state)
            )
        return integrated_fluxes, original_fluxes

    def recompute(
        self,
        states,
        dg_candidate,
        dt,
        troubled,
        boundary_state=None,
        additional_rhs=None,
    ):
        """Replace troubled Euler candidates and correct adjacent DG means."""
        subcell_states = self.project(states)
        residual = self._internal_residual(subcell_states)
        new_fluxes, old_fluxes = self._external_fluxes(
            states, subcell_states, troubled, residual, boundary_state
        )
        fv_candidate = {
            name: subcell_states[name]
            + dt * residual[name] / self.projections[name].volumes[..., None]
            for name in subcell_states
        }
        if additional_rhs is not None:
            fv_candidate = {
                name: value
                + dt * dg_to_subcells(additional_rhs[name], self.projections[name])
                for name, value in fv_candidate.items()
            }
        reconstructed = {
            name: subcells_to_dg(value, self.projections[name])
            for name, value in fv_candidate.items()
        }
        corrected = dict(dg_candidate)
        for group, new_flux, old_flux in zip(
            self.face_groups, new_fluxes, old_fluxes
        ):
            faces = self.mesh.face_blocks[group["face_type"]]
            indices = group["indices"]
            owner_block = self.mesh.element_blocks[group["owner_block"]]
            owner_name = owner_block.element_type
            owner_cells = jnp.asarray(faces.owner_local)[indices]
            owner_flag = troubled[owner_name][owner_cells]
            if group["neighbor_block"] >= 0:
                neighbor_block = self.mesh.element_blocks[group["neighbor_block"]]
                neighbor_name = neighbor_block.element_type
                neighbor_cells = jnp.asarray(faces.neighbor_local)[indices]
                neighbor_flag = troubled[neighbor_name][neighbor_cells]
            else:
                neighbor_name = None
                neighbor_cells = None
                neighbor_flag = jnp.zeros_like(owner_flag)
            active = owner_flag | neighbor_flag
            difference = jnp.where(
                active[:, None], new_flux - old_flux, 0.0
            )
            owner_volume = jnp.sum(
                self.projections[owner_name].volumes[owner_cells], axis=1
            )
            owner_correction = -dt * difference / owner_volume[:, None]
            owner_correction = jnp.where(
                (active & ~owner_flag)[:, None], owner_correction, 0.0
            )
            corrected[owner_name] = corrected[owner_name].at[owner_cells].add(
                owner_correction[:, None, :]
            )
            if neighbor_name is not None:
                neighbor_volume = jnp.sum(
                    self.projections[neighbor_name].volumes[neighbor_cells], axis=1
                )
                neighbor_correction = dt * difference / neighbor_volume[:, None]
                neighbor_correction = jnp.where(
                    (active & ~neighbor_flag)[:, None], neighbor_correction, 0.0
                )
                corrected[neighbor_name] = corrected[neighbor_name].at[
                    neighbor_cells
                ].add(neighbor_correction[:, None, :])

        return {
            name: jnp.where(
                troubled[name][:, None, None], reconstructed[name], corrected[name]
            )
            for name in states
        }
