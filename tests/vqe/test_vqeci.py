# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import pytest
from openfermion import InteractionOperator, MolecularData
from openfermionpyscf import run_pyscf
from pyscf import gto
from quri_parts.algo.ansatz import HardwareEfficient, SymmetryPreserving
from quri_parts.chem.ansatz import (
    AllSinglesDoubles,
    GateFabric,
    ParticleConservingU1,
    ParticleConservingU2,
)
from quri_parts.core.state import (
    ComputationalBasisState,
    comp_basis_superposition,
)
from quri_parts.openfermion.ansatz import KUpCCGSD, TrotterUCCSD
from quri_parts.openfermion.transforms import bravyi_kitaev, jordan_wigner

from chemqulacs.util import utils
from chemqulacs.vqe.vqeci import (
    VQECI,
    Ansatz,
    _create_ansatz,
    _get_active_hamiltonian,
)


def vqeci() -> VQECI:
    n_qubits = 8
    n_electrons = 4

    # HF state
    bits = 0
    for i in range(n_electrons):
        bits += 1 << i
    state = ComputationalBasisState(n_qubits, bits=bits)

    # Dummy mol.
    geom_water = utils.get_geometry_from_pubchem("water")
    mol = gto.M(atom=geom_water, basis="sto-3g")

    vqe_ci = VQECI(mol=mol)
    vqe_ci.initial_states = [state]
    vqe_ci.opt_states = [state]
    vqe_ci.n_orbitals = int(n_qubits / 2)
    vqe_ci.n_qubit = n_qubits

    return vqe_ci


def test_get_active_hamiltonian():
    n_orbitals = 2
    e_core = 1.0

    h1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    h2 = np.array(
        [
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
        ]
    )

    one_body_tensor = np.array(
        [
            [0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.0, 0.2],
            [0.3, 0.0, 0.4, 0.0],
            [0.0, 0.3, 0.0, 0.4],
        ]
    )
    two_body_tensor = np.array(
        [
            [
                [
                    [0.05, 0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.05, 0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.0],
                ],
                [
                    [0.15, 0.0, 0.35, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.4, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.15, 0.0, 0.35, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.4, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.05, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.3],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.05, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.3],
                ],
                [
                    [0.0, 0.15, 0.0, 0.35],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.4],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.15, 0.0, 0.35],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.4],
                ],
            ],
            [
                [
                    [0.05, 0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.05, 0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.0],
                ],
                [
                    [0.15, 0.0, 0.35, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.4, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.15, 0.0, 0.35, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.4, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.05, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.3],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.05, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.3],
                ],
                [
                    [0.0, 0.15, 0.0, 0.35],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.4],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.15, 0.0, 0.35],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.4],
                ],
            ],
        ]
    )

    assert _get_active_hamiltonian(h1, h2, n_orbitals, e_core) == InteractionOperator(
        e_core, one_body_tensor, two_body_tensor
    )

    # H2 molecule, CAS(2,2)
    n_orbitals = 2
    e_core = 0.70556961456
    h1 = np.array([[-1.24728451, 0.0], [0.0, -0.481272931]])
    h2 = np.array(
        [
            [0.672847947, 0.0, 0.661977259],
            [0.0, 0.181771537, 0.0],
            [0.661977259, 0.0, 0.695815151],
        ]
    )
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.75))]

    pyscf_h2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(pyscf_h2_molecule)

    assert _get_active_hamiltonian(
        h1, h2, n_orbitals, e_core
    ) == molecule.get_molecular_hamiltonian([], [0, 1])

    # LiH molecule, CAS(2,2)
    n_orbitals = 2
    e_core = -6.8179118506205585
    h1 = np.array([[-0.7617059, 0.0497932], [0.0497932, -0.3520365]])
    h2 = np.array(
        [
            [0.48247702, -0.04979318, 0.22169206],
            [-0.04979318, 0.01380829, 0.00840803],
            [0.22169206, 0.00840803, 0.3370786],
        ]
    )
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    geometry = [("Li", (0, 0, 0)), ("H", (0, 0, 1.67))]

    pyscf_h2_molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(pyscf_h2_molecule)

    assert _get_active_hamiltonian(
        h1, h2, n_orbitals, e_core
    ) == molecule.get_molecular_hamiltonian([0], [1, 2])


def test_make_rdm1():
    vqe_ci = vqeci()
    assert np.allclose(
        vqe_ci.make_rdm1(None, vqe_ci.n_orbitals, [2, 2]),
        np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )

    vqe_ci.opt_states = [
        comp_basis_superposition(
            vqe_ci.opt_states[0],
            ComputationalBasisState(8, bits=0b11110000),
            np.pi / 4,
            0,
        )
    ]
    assert np.allclose(
        vqe_ci.make_rdm1(None, vqe_ci.n_orbitals, [2, 2]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )


def test_make_rdm12():
    vqe_ci = vqeci()
    _, rdm2 = vqe_ci.make_rdm12(None, vqe_ci.n_orbitals, [2, 2])
    expected = np.array(
        [
            [
                [[2, 0, 0, 0], [0, 4, -0, -0], [0, -0, -0, -0], [0, -0, -0, -0]],
                [[0, 0, 0, 0], [-2, 0, -0, -0], [0, 0, -0, -0], [0, 0, -0, -0]],
                [[0, 0, 0, 0], [0, 0, 0, -0], [0, 0, 0, -0], [0, 0, 0, -0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, -2, 0, 0], [0, 0, 0, 0], [0, -0, -0, -0], [0, -0, -0, -0]],
                [[4, 0, 0, 0], [0, 2, 0, 0], [0, 0, -0, -0], [0, 0, -0, -0]],
                [[-0, -0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -0], [0, 0, 0, -0]],
                [[-0, -0, -0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, -0, -0, -0]],
                [[-0, 0, 0, 0], [-0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -0, -0]],
                [[-0, -0, 0, 0], [-0, -0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -0]],
                [[-0, -0, -0, 0], [-0, -0, -0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[-0, 0, 0, 0], [-0, 0, 0, 0], [-0, 0, 0, 0], [0, 0, 0, 0]],
                [[-0, -0, 0, 0], [-0, -0, 0, 0], [-0, -0, 0, 0], [0, 0, 0, 0]],
                [[-0, -0, -0, 0], [-0, -0, -0, 0], [-0, -0, -0, 0], [0, 0, 0, 0]],
            ],
        ]
    )
    assert np.allclose(rdm2, expected)

    vqe_ci.opt_states = [
        comp_basis_superposition(
            vqe_ci.opt_states[0],
            ComputationalBasisState(8, bits=0b11110000),
            np.pi / 4,
            0,
        )
    ]
    _, rdm2 = vqe_ci.make_rdm12(None, vqe_ci.n_orbitals, [2, 2])
    expected = np.array(
        [
            [
                [[1, 0, 0, 0], [0, 2, -0, -0], [0, -0, -0, -0], [0, -0, -0, -0]],
                [[0, 0, 0, 0], [-1, 0, -0, -0], [0, 0, -0, -0], [0, 0, -0, -0]],
                [[0, 0, 0, 0], [0, 0, 0, -0], [0, 0, 0, -0], [0, 0, 0, -0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, -1, 0, 0], [0, 0, 0, 0], [0, -0, -0, -0], [0, -0, -0, -0]],
                [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, -0, -0], [0, 0, -0, -0]],
                [[-0, -0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -0], [0, 0, 0, -0]],
                [[-0, -0, -0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, -0, -0, -0]],
                [[-0, 0, 0, 0], [-0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -0, -0]],
                [[-0, -0, 0, 0], [-0, -0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]],
                [[-0, -0, -0, 0], [-0, -0, -0, 0], [0, 0, 0, 0], [0, 0, -1, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[-0, 0, 0, 0], [-0, 0, 0, 0], [-0, 0, 0, 0], [0, 0, 0, 0]],
                [[-0, -0, 0, 0], [-0, -0, 0, 0], [-0, -0, 0, -1], [0, 0, 0, 0]],
                [[-0, -0, -0, 0], [-0, -0, -0, 0], [-0, -0, 2, 0], [0, 0, 0, 1]],
            ],
        ]
    )
    assert np.allclose(rdm2, expected)


def test_make_dm2():
    vqe_ci = vqeci()
    dm2 = vqe_ci.make_dm2(None, vqe_ci.n_orbitals,2)
    expected = np.array(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, -2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 2, 0, 0], [-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )
    assert np.allclose(dm2, expected)

    vqe_ci.opt_states = [
        comp_basis_superposition(
            vqe_ci.opt_states[0],
            ComputationalBasisState(8, bits=0b11110000),
            np.pi / 4,
            0,
        )
    ]
    dm2 = vqe_ci.make_dm2(None, vqe_ci.n_orbitals,2)
    expected = np.array(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )
    assert np.allclose(dm2, expected)


class TestCreateAnsatz:
    def default_args(self) -> dict[str, Any]:
        args = {
            "n_sorbs": 4,
            "n_electrons": 2,
            "fermion_qubit_mapping": jordan_wigner,
            "layers": 2,
            "k": 1,
            "trotter_number": 1,
            "include_pi": False,
            "use_singles": True,
            "delta_sz": False,
            "singlet_excitation": False,
        }
        return args

    def test_hwe(self) -> None:
        args = self.default_args()

        hwe_default = _create_ansatz(Ansatz.HardwareEfficient, **args)
        assert type(hwe_default) == HardwareEfficient
        assert len(hwe_default.gates) == 30

        args["layers"] = 1
        hwe_l1 = _create_ansatz(Ansatz.HardwareEfficient, **args)
        assert type(hwe_l1) == HardwareEfficient
        assert len(hwe_l1.gates) == 19

    def test_sp(self) -> None:
        args = self.default_args()

        sp_default = _create_ansatz(Ansatz.SymmetryPreserving, **args)
        assert type(sp_default) == SymmetryPreserving
        assert len(sp_default.gates) == 42

        args["layers"] = 1
        sp_l1 = _create_ansatz(Ansatz.SymmetryPreserving, **args)
        assert type(sp_l1) == SymmetryPreserving
        assert len(sp_l1.gates) == 21

    def test_all_singles_doubles(self) -> None:
        args = self.default_args()

        asd_default = _create_ansatz(Ansatz.AllSinglesDoubles, **args)
        assert type(asd_default) == AllSinglesDoubles
        assert len(asd_default.gates) == 40

        args["n_electrons"] = 3
        asd_elec3 = _create_ansatz(Ansatz.AllSinglesDoubles, **args)
        assert type(asd_elec3) == AllSinglesDoubles
        assert len(asd_elec3.gates) == 6

    def test_particle_conserving_u1(self) -> None:
        args = self.default_args()

        pu1_default = _create_ansatz(Ansatz.ParticleConservingU1, **args)
        assert type(pu1_default) == ParticleConservingU1
        assert len(pu1_default.gates) == 174

        args["layers"] = 1
        pu1_l1 = _create_ansatz(Ansatz.ParticleConservingU1, **args)
        assert type(pu1_l1) == ParticleConservingU1
        assert len(pu1_l1.gates) == 87

    def test_particle_conserving_u2(self) -> None:
        args = self.default_args()

        pu2_default = _create_ansatz(Ansatz.ParticleConservingU2, **args)
        assert type(pu2_default) == ParticleConservingU2
        assert len(pu2_default.gates) == 56

        args["layers"] = 1
        pu2_l1 = _create_ansatz(Ansatz.ParticleConservingU2, **args)
        assert type(pu2_l1) == ParticleConservingU2
        assert len(pu2_l1.gates) == 28

    def test_gate_fabric(self) -> None:
        args = self.default_args()

        gf_default = _create_ansatz(Ansatz.GateFabric, **args)
        assert type(gf_default) == GateFabric
        assert len(gf_default.gates) == 80

        args["layers"] = 1
        gf_l1 = _create_ansatz(Ansatz.GateFabric, **args)
        assert type(gf_l1) == GateFabric
        assert len(gf_l1.gates) == 40

        args["include_pi"] = True
        gf_l1_pi = _create_ansatz(Ansatz.GateFabric, **args)
        assert type(gf_l1_pi) == GateFabric
        assert len(gf_l1_pi.gates) == 52

    def test_uccsd(self) -> None:
        default = self.default_args()

        uccsd_default = _create_ansatz(Ansatz.UCCSD, **default)
        assert type(uccsd_default) == TrotterUCCSD
        assert len(uccsd_default.gates) == 12

        args = default.copy()
        args["trotter_number"] = 2
        uccsd_trotter2 = _create_ansatz(Ansatz.UCCSD, **args)
        assert type(uccsd_trotter2) == TrotterUCCSD
        assert len(uccsd_trotter2.gates) == 24

        args = default.copy()
        args["fermion_qubit_mapping"] = bravyi_kitaev
        uccsd_bk = _create_ansatz(Ansatz.UCCSD, **args)
        assert type(uccsd_bk) == TrotterUCCSD
        assert len(uccsd_bk.gates) == 12
        assert uccsd_default.gates[0] != uccsd_bk.gates[0]

        args = default.copy()
        args["use_singles"] = False
        uccd = _create_ansatz(Ansatz.UCCSD, **args)
        assert type(uccd) == TrotterUCCSD
        assert len(uccd.gates) == 8

        args = default.copy()
        args["singlet_excitation"] = True
        uccsd_singlet = _create_ansatz(Ansatz.UCCSD, **args)
        assert type(uccsd_singlet) == TrotterUCCSD
        assert len(uccsd_singlet.gates) == 12
        assert uccsd_singlet.parameter_count != uccsd_default.parameter_count

        args = default.copy()
        args["singlet_excitation"] = True
        args["delta_sz"] = 1
        with pytest.raises(ValueError):
            _create_ansatz(Ansatz.UCCSD, **args)

    def test_kupccgsd(self) -> None:
        default = self.default_args()

        kupccgsd_default = _create_ansatz(Ansatz.KUpCCGSD, **default)
        assert type(kupccgsd_default) == KUpCCGSD
        assert len(kupccgsd_default.gates) == 12

        args = default.copy()
        args["trotter_number"] = 2
        kupccgsd_trotter2 = _create_ansatz(Ansatz.KUpCCGSD, **args)
        assert type(kupccgsd_trotter2) == KUpCCGSD
        assert len(kupccgsd_trotter2.gates) == 24

        args = default.copy()
        args["fermion_qubit_mapping"] = bravyi_kitaev
        kupccgsd_bk = _create_ansatz(Ansatz.KUpCCGSD, **args)
        assert type(kupccgsd_bk) == KUpCCGSD
        assert len(kupccgsd_bk.gates) == 12
        assert kupccgsd_default.gates[0] != kupccgsd_bk.gates[0]

        args = default.copy()
        args["k"] = 2
        kupccgsd_k2 = _create_ansatz(Ansatz.KUpCCGSD, **args)
        assert type(kupccgsd_k2) == KUpCCGSD
        assert len(kupccgsd_k2.gates) == 24

        args = default.copy()
        args["singlet_excitation"] = True
        kupccgsd_singlet = _create_ansatz(Ansatz.KUpCCGSD, **args)
        assert type(kupccgsd_singlet) == KUpCCGSD
        assert len(kupccgsd_singlet.gates) == 12
        assert kupccgsd_singlet.parameter_count != kupccgsd_default.parameter_count
