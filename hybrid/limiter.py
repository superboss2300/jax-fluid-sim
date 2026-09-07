"""Conservative positivity and troubled-element detection for hybrid DG."""

import jax
import jax.numpy as jnp
import numpy as np

from .elements import p1_basis
from .subcell import HybridSubcellFV


class HybridPositivityLimiter:
    """Zhang--Shu-style scaling toward each physical cell average."""

    def __init__(self, operator, density_floor=1.0e-10, pressure_floor=1.0e-10):
        self.operator = operator
        self.solver = operator.solver
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.face_sample_basis = self._build_face_sample_basis()

    def _build_face_sample_basis(self):
        """Basis values at every distinct face quadrature point.

        Positivity at only the interpolation nodes and volume quadrature points
        does not guarantee positivity at the flux quadrature points.  Gather
        the small, element-type-specific set once instead of rebuilding it in
        every limiter call.
        """
        points = {
            block.element_type: [] for block in self.solver.mesh.element_blocks
        }
        for face_type, owner_block, neighbor_block, indices in self.operator.face_groups:
            metrics = self.operator.geometry.face_metrics[face_type]
            owner_type = self.solver.mesh.element_blocks[owner_block].element_type
            points[owner_type].append(
                np.asarray(metrics.owner_reference_points)[np.asarray(indices)]
                .reshape(-1, 3)
            )
            if neighbor_block >= 0:
                neighbor_type = self.solver.mesh.element_blocks[
                    neighbor_block
                ].element_type
                points[neighbor_type].append(
                    np.asarray(metrics.neighbor_reference_points)[np.asarray(indices)]
                    .reshape(-1, 3)
                )

        result = {}
        for element_type, chunks in points.items():
            reference = self.operator.geometry.reference_elements[element_type]
            if chunks:
                unique = np.unique(
                    np.round(np.concatenate(chunks, axis=0), decimals=12), axis=0
                )
                result[element_type] = reference.basis(jnp.asarray(unique))
            else:
                result[element_type] = jnp.empty(
                    (0, len(reference.nodes)), dtype=reference.nodes.dtype
                )
        return result

    def _mean(self, block_id, U):
        block = self.solver.mesh.element_blocks[block_id]
        reference = self.operator.geometry.reference_elements[block.element_type]
        metrics = self.operator.geometry.element_metrics[block_id]
        Uq = jnp.einsum(
            "qi,eiv->eqv", reference.basis(reference.volume_points), U,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.einsum(
            "eqv,eq->ev", Uq, metrics.integration_weights,
            precision=jax.lax.Precision.HIGHEST,
        ) / jnp.sum(metrics.integration_weights, axis=1)[:, None]

    def _sample(self, element_type, U):
        reference = self.operator.geometry.reference_elements[element_type]
        Uq = jnp.einsum(
            "qi,eiv->eqv", reference.basis(reference.volume_points), U,
            precision=jax.lax.Precision.HIGHEST,
        )
        Uf = jnp.einsum(
            "qi,eiv->eqv", self.face_sample_basis[element_type], U,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.concatenate([U, Uq, Uf], axis=1)

    def _pressure_ok(self, mean, deviations, theta):
        samples = mean[:, None, :] + theta[:, None, None] * deviations
        pressure = self.solver.conserved_to_primitive(samples)[..., 4]
        return jnp.min(pressure, axis=1) >= self.pressure_floor

    def limit_block(self, block_id, U):
        block = self.solver.mesh.element_blocks[block_id]
        mean = self._mean(block_id, U)
        samples = self._sample(block.element_type, U)
        deviations = samples - mean[:, None, :]
        minimum_density = jnp.min(samples[..., 0], axis=1)
        density_denominator = mean[:, 0] - minimum_density
        density_theta = jnp.where(
            minimum_density < self.density_floor,
            (mean[:, 0] - self.density_floor)
            / jnp.maximum(density_denominator, 1.0e-30),
            1.0,
        )
        upper = jnp.clip(density_theta, 0.0, 1.0)
        lower = jnp.zeros_like(upper)

        def bisect(_, interval):
            low, high = interval
            middle = 0.5 * (low + high)
            valid = self._pressure_ok(mean, deviations, middle)
            return jnp.where(valid, middle, low), jnp.where(valid, high, middle)

        lower, _ = jax.lax.fori_loop(0, 30, bisect, (lower, upper))
        pressure_at_upper_ok = self._pressure_ok(mean, deviations, upper)
        theta = jnp.where(pressure_at_upper_ok, upper, lower)
        return mean[:, None, :] + theta[:, None, None] * (
            U - mean[:, None, :]
        )

    def __call__(self, states):
        return {
            block.element_type: self.limit_block(
                block_id, states[block.element_type]
            )
            for block_id, block in enumerate(self.solver.mesh.element_blocks)
        }


class HybridTroubledCellSensor:
    """Persson-like P2 high-mode sensor using density and pressure."""

    def __init__(self, operator, threshold=5.0e-2):
        self.operator = operator
        self.solver = operator.solver
        self.threshold = float(threshold)
        self.p1_projectors = {}
        if self.solver.polynomial_degree >= 2:
            for element_type, reference in operator.geometry.reference_elements.items():
                p1_values = p1_basis(element_type, reference.nodes)
                self.p1_projectors[element_type] = (
                    p1_values @ jnp.linalg.pinv(p1_values)
                )

    def block_sensor(self, element_type, U):
        if self.solver.polynomial_degree < 2:
            return jnp.zeros(U.shape[0], dtype=U.dtype)
        primitive = self.solver.conserved_to_primitive(U)
        monitored = primitive[..., jnp.asarray([0, 4])]
        low = jnp.einsum(
            "ij,ejm->eim", self.p1_projectors[element_type], monitored,
            precision=jax.lax.Precision.HIGHEST,
        )
        high_energy = jnp.sum((monitored - low)**2, axis=1)
        scale_energy = jnp.sum(monitored**2, axis=1)
        # Normalized Persson modal shock indicator: high-mode energy relative to total element state
        ratio = high_energy / jnp.maximum(scale_energy, 1.0e-12)
        return jnp.max(ratio, axis=-1)

    def __call__(self, states):
        sensor = {
            name: self.block_sensor(name, U) for name, U in states.items()
        }
        troubled = {name: value > self.threshold for name, value in sensor.items()}
        return sensor, troubled


class HybridModalShockLimiter:
    """Conservative P2-to-P1 fallback in sensor-selected elements.

    This is the robust hierarchical fallback used until the a-posteriori
    subcell FV recomputation is complete. It preserves linear content and the
    physical cell average; it does not collapse a troubled element to P0.
    """

    def __init__(
        self,
        operator,
        threshold=5.0e-2,
        positivity=None,
        p0_fallback_factor=100.0,
        p0_relative_floor=0.2,
    ):
        self.operator = operator
        self.sensor = HybridTroubledCellSensor(operator, threshold)
        self.positivity = (
            HybridPositivityLimiter(operator) if positivity is None else positivity
        )
        self.p0_fallback_factor = float(p0_fallback_factor)
        self.p0_relative_floor = float(p0_relative_floor)

    def __call__(self, states):
        if self.operator.solver.polynomial_degree < 2:
            return self.positivity(states)
        _, troubled = self.sensor(states)
        limited = {}
        for block_id, block in enumerate(self.operator.mesh.element_blocks):
            name = block.element_type
            U = states[name]
            projector = self.sensor.p1_projectors[name]
            low = jnp.einsum(
                "ij,ejv->eiv", projector, U,
                precision=jax.lax.Precision.HIGHEST,
            )
            old_mean = self.positivity._mean(block_id, U)
            low_mean = self.positivity._mean(block_id, low)
            low = low + (old_mean - low_mean)[:, None, :]
            # P1 can still overshoot at a strong shock.  If its reconstructed
            # density or pressure is invalid (or extremely close to the
            # positivity floor), fall back to the conservative P0 mean for
            # that element.  Smooth and ordinarily troubled elements retain
            # P2 and P1 respectively.
            low_samples = self.positivity._sample(name, low)
            low_primitive = self.operator.solver.conserved_to_primitive(
                low_samples
            )
            density_min = jnp.min(low_primitive[..., 0], axis=1)
            pressure_min = jnp.min(low_primitive[..., 4], axis=1)
            mean_primitive = self.operator.solver.conserved_to_primitive(old_mean)
            mean_density = jnp.maximum(
                mean_primitive[..., 0], self.positivity.density_floor
            )
            mean_pressure = jnp.maximum(
                mean_primitive[..., 4], self.positivity.pressure_floor
            )
            finite = jnp.all(jnp.isfinite(low_samples), axis=(1, 2))
            severe = (
                (~finite)
                | (
                    density_min
                    < self.p0_fallback_factor * self.positivity.density_floor
                )
                | (
                    pressure_min
                    < self.p0_fallback_factor * self.positivity.pressure_floor
                )
                | (density_min < self.p0_relative_floor * mean_density)
                | (pressure_min < self.p0_relative_floor * mean_pressure)
            )
            fallback = jnp.where(severe[:, None, None], old_mean[:, None, :], low)
            limited[name] = jnp.where(
                troubled[name][:, None, None], fallback, U
            )
        return self.positivity(limited)


class HybridMOODSubcellLimiter:
    """A-posteriori P2-DG/subcell-FV recomputation.

    The unlimited DG candidate supplies detection only.  Flagged elements are
    recomputed from the pre-stage state on conforming finite-volume subcells;
    neighboring DG means receive the equal-and-opposite interface-flux
    correction.  Positivity scaling is applied only after that conservative
    recomputation.
    """

    def __init__(self, operator, threshold=5.0e-2, positivity=None, reconstruction="weno"):
        self.operator = operator
        self.sensor = HybridTroubledCellSensor(operator, threshold)
        self.positivity = (
            HybridPositivityLimiter(operator) if positivity is None else positivity
        )
        self.subcell = HybridSubcellFV(operator, reconstruction=reconstruction)

    def troubled(self, candidate):
        _, flags = self.sensor(candidate)
        result = {}
        for block_id, block in enumerate(self.operator.mesh.element_blocks):
            name = block.element_type
            samples = self.positivity._sample(name, candidate[name])
            mean = self.positivity._mean(block_id, candidate[name])
            samples = jnp.concatenate([samples, mean[:, None, :]], axis=1)
            primitive = self.operator.solver.conserved_to_primitive(samples)
            invalid = (
                ~jnp.all(jnp.isfinite(primitive), axis=(1, 2))
                | (jnp.min(primitive[..., 0], axis=1) < self.positivity.density_floor)
                | (jnp.min(primitive[..., 4], axis=1) < self.positivity.pressure_floor)
            )
            result[name] = flags[name] | invalid
        return result

    def recompute(
        self,
        base,
        dg_candidate,
        dt,
        boundary_state=None,
        additional_rhs=None,
        detection_state=None,
    ):
        troubled = self.troubled(
            dg_candidate if detection_state is None else detection_state
        )
        any_troubled = jnp.asarray(False)
        for flag in troubled.values():
            any_troubled = any_troubled | jnp.any(flag)

        def perform_recompute():
            return self.subcell.recompute(
                base,
                dg_candidate,
                dt,
                troubled,
                boundary_state,
                additional_rhs,
            )

        corrected = jax.lax.cond(
            any_troubled,
            perform_recompute,
            lambda: dg_candidate,
        )
        return self.positivity(corrected)

    def __call__(self, states):
        # Ordinary calls must not invent a time update.  SSP-RK invokes
        # recompute() with the pre-stage state and timestep.
        return self.positivity(states)
