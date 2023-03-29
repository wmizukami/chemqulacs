# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from openfermion import InteractionOperator, MolecularData
from openfermionpyscf import run_pyscf
from pyscf import gto
from quri_parts.core.state import (
    ComputationalBasisState,
    comp_basis_superposition,
)

from chemqulacs.util import utils
from chemqulacs.vqe.vqeci import VQECI, _get_active_hamiltonian


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
    vqe_ci.initial_state = state
    vqe_ci.opt_state = state
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

    vqe_ci.opt_state = comp_basis_superposition(
        vqe_ci.opt_state, ComputationalBasisState(8, bits=0b11110000), np.pi / 4, 0
    )
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

    vqe_ci.opt_state = comp_basis_superposition(
        vqe_ci.opt_state, ComputationalBasisState(8, bits=0b11110000), np.pi / 4, 0
    )
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
    dm2 = vqe_ci.make_dm2(None, vqe_ci.n_orbitals, [2, 2])
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

    vqe_ci.opt_state = comp_basis_superposition(
        vqe_ci.opt_state, ComputationalBasisState(8, bits=0b11110000), np.pi / 4, 0
    )
    dm2 = vqe_ci.make_dm2(None, vqe_ci.n_orbitals, [2, 2])
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
