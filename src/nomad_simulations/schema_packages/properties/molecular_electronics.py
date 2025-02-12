import numpy as np
from nomad.config import config
from nomad.metainfo import (
    Quantity,
    SchemaPackage,
    Section,
    SubSection,
    MEnum,
)
from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
from ..base_sections import ModelBaseSection
from .common_properties import energy
from ..atoms_state import OrbitalsState
import plotly.graph_objects as go

configuration = config.get_plugin_entry_point(
    'nomad_simulations.schema_packages:nomad_simulations_plugin'
)

m_package = SchemaPackage()


# ? split out spin for better handling
SingleElectronSimpleSpin = Quantity(
    type=MEnum('alpha', 'beta'),
    default='alpha',
    description='Simple spin',
)


class ProjectionTarget(ModelBaseSection, OrbitalsState):
    """Label section for an element, ionic state, or (atomic) quantum state"""

    element = Quantity(
        type=str,
    )

    atom_index = Quantity(
        type=int,
        shape=['*'],
    )

    def name_from_section(self) -> str:
        projected = False
        name = ''
        if self.element is not None:
            name += self.element
            projected = True
        if self.atom_index is not None:
            name += f'_{self.atom_index}'
            projected = True
        if self.l_quantum_symbol is not None:
            projected = True
            if self.n_quantum_number is not None:
                name += f' {self.n_quantum_number}{self.l_quantum_symbol}'
            else:
                name += f' {self.l_quantum_symbol}'
        return name if projected else 'total'


class SemanticGroup(ModelBaseSection):  # ! hide from metainfo
    """
    Abstract interface for a generic kind of grouping.
    It produces a scatter framework marked by `label`.
    """

    label = None

    def name_from_section(self) -> str:  # !
        return self.label.name_from_section()

    def plot(self) -> go.Scatter:
        """Generate an individual plotly plot."""
        pass


class SemanticGroupContainer(ModelBaseSection, PlotSection):
    """Container for semantic groups of electronic states"""  # ! re-word

    m_def = Section()

    groups = SubSection(sub_section=SemanticGroup.m_def, repeats=True)

    def plot(self) -> PlotlyFigure:
        figure = go.Figure()
        for group in self.groups:
            figure.add_trace(group.plot())
        return PlotlyFigure(figure=figure.to_plotly_json())

    def normalize(self, *args, **kwargs) -> None:
        super(ModelBaseSection, self).normalize(*args, **kwargs)
        super(PlotSection, self).normalize(*args, **kwargs)


class Frontiers(ModelBaseSection):
    """Frontier orbitals of the electronic states."""

    homo = Quantity(
        type=np.float64,
        unit='joule',
        description='Highest occupied molecular orbital',
    )

    lumo = Quantity(
        type=np.float64,
        unit='joule',
        description='Lowest unoccupied molecular orbital',
    )

    energy_gap = Quantity(
        type=np.float64,
        unit='joule',
        description='The energy gap between the homo and lumo',
    )


class ElectronicEigenvalues(SemanticGroupContainer):
    """Eigenvalues of the electronic states."""

    class EigenvalueGroup(SemanticGroupContainer):
        class EigenvalueLabel(ProjectionTarget):  # ? necessary
            spin = SingleElectronSimpleSpin

        label = SubSection(sub_section=EigenvalueLabel.m_def)

        energies = Quantity(
            type=np.float64,
            unit='joule',
            shape=['*'],
            description='The eigenstate obtained from solving the electronic Schrödinger equation',  # ! re-word
        )

        occupations = Quantity(
            type=np.float64,  # ! use typing for better restrictions
            shape=['*'],
            description='Occupation of the states',
        )

    groups = SubSection(sub_section=EigenvalueGroup.m_def, repeats=True)


class MolecularOrbitals(ModelBaseSection):
    """
    This class stores all molecular orbitals (MO) in a single container, with each Quantity
    indexed by `mo_num` and (where applicable) `ao_num`.

    It captures data analogous to the TREXIO specification, for example:
      - mo/type           -> mo_type
      - mo/num            -> mo_num
      - mo/coefficient    -> coefficient
      - mo/coefficient_im -> coefficient_im
      - mo/symmetry       -> symmetry
      - mo/occupation     -> occupation
      - mo/energy         -> energy
      - mo/spin           -> spin

    Additionally, the frontier orbitals (HOMO, LUMO, energy gap) are stored as a sub-section.
    """

    mo_num = Quantity(
        type=np.int32,
        description='Number of molecular orbitals.',
    )

    ao_num = Quantity(
        type=np.int32,
        description="""
        Number of atomic orbitals or basis functions (often needed for coefficient shape).
        Corresponds to the 'ao.num' dimension in TREXIO.
        """,
    )

    mo_type = Quantity(
        type=str,
        shape=['mo_num'],
        description="""
        Type of each molecular orbital, e.g. "canonical", "localized", or "active".
        """
    )

    coefficient = Quantity(
        type=np.float64,
        shape=['mo_num', 'ao_num'],
        description="""
        Real part of the MO coefficients, shape [mo_num, ao_num].
        Each row corresponds to one MO, each column to a basis function/atomic orbital.
        """
    )

    coefficient_im = Quantity(
        type=np.float64,
        shape=['mo_num', 'ao_num'],
        description="""
        Imaginary part of the MO coefficients, shape [mo_num, ao_num].
        Can be zero if the orbitals are purely real.
        """
    )

    symmetry = Quantity(
        type=str,
        shape=['mo_num'],
        description="""
        Symmetry label for each MO (e.g. group-theory labels like A1, B2, or simpler "sigma"/"pi").
        """
    )

    occupation = Quantity(
        type=np.float64,
        shape=['mo_num'],
        description="""
        Occupation numbers for each MO (commonly in [0,2] for closed-shell, but may be
        fractional for open-shell or multi-reference methods).
        """
    )

    energy = Quantity(
        type=np.float64,
        shape=['mo_num'],
        unit='joule',
        description="""
        Orbital energies for each MO (in joules). These can be converted from eV or Hartree
        as needed.
        """
    )

    spin = Quantity(
        type=SingleElectronSimpleSpin.type,  # MEnum('alpha', 'beta')
        shape=['mo_num'],
        description="""
        Spin channel for each MO if this is an unrestricted open-shell set.
        "alpha" or "beta" for each orbital. For restricted/closed-shell systems, all may be "alpha".
        """
    )

    frontiers = SubSection(
        sub_section=Frontiers.m_def,
        description="Frontier orbitals: HOMO, LUMO, and energy gap."
    )

    def normalize(self, *args, **kwargs) -> None:
        """
        Normalize the sub-section and, if necessary, derive the frontier values from
        the orbital data. If the frontiers (HOMO, LUMO, energy gap) are not provided,
        they are computed from the occupations and energies.
        """
        super().normalize(*args, **kwargs)
        if not self.frontiers:
            self.m_setdefault('frontiers')
        # Ensure that we have occupation and energy data available
        if self.occupation is not None and self.energy is not None:
            occ = np.array(self.occupation)
            en = np.array(self.energy)
            # Derive HOMO: maximum energy among orbitals with occupation > 0
            if self.frontiers.homo is None:
                occupied = np.where(occ > 0.0)[0]
                if occupied.size > 0:
                    self.frontiers.homo = float(np.max(en[occupied]))
            # Derive LUMO: minimum energy among orbitals with zero occupation
            if self.frontiers.lumo is None:
                unoccupied = np.where(occ == 0.0)[0]
                if unoccupied.size > 0:
                    self.frontiers.lumo = float(np.min(en[unoccupied]))
            # Compute energy gap if both HOMO and LUMO are defined
            if (self.frontiers.homo is not None and self.frontiers.lumo is not None
                    and self.frontiers.energy_gap is None):
                self.frontiers.energy_gap = self.frontiers.lumo - self.frontiers.homo


m_package.__init_metainfo__()
