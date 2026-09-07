"""Conservative-state layout shared by all hybrid element types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressibleStateLayout:
    """Describe Euler/Navier--Stokes plus optional reacting-species storage.

    The first five entries never change, so inviscid and viscous flux code can
    remain independent of the chemistry mechanism. Species are conservative
    partial densities rho*Y_s. Additional nonequilibrium energies can be added
    after the species block later without changing mesh connectivity.
    """

    species_names: tuple[str, ...] = ()

    density: int = 0
    momentum: slice = slice(1, 4)
    total_energy: int = 4

    @property
    def species(self) -> slice:
        return slice(5, 5 + len(self.species_names))

    @property
    def number_of_variables(self) -> int:
        return 5 + len(self.species_names)

    def index(self, name: str) -> int:
        try:
            return 5 + self.species_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown species {name!r}") from error
