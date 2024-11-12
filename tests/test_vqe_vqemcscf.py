# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf import gto, mcscf, scf
from quri_parts.algo.optimizer import LBFGS
from quri_parts.core.state import comp_basis_superposition

from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf
from chemqulacs.vqe.vqeci import Ansatz, _generate_inital_states

from math import pi


geom_water = utils.get_geometry_from_pubchem("water")
mol = gto.M(atom=geom_water, basis="sto-3g")
mf = scf.RHF(mol)
mf.run()


# def test_vqecasci():
#   mc = vqemcscf.VQECASCI(mf, 2, 2)


# def test_vqecasci():
#   mc = vqemcscf.VQECASSCF(mf, 2, 2)


def test_vqecasci_h2o_2e_2o():
    mc = vqemcscf.VQECASCI(
        mf, 2, 2, optimizer=LBFGS(), ansatz=Ansatz.GateFabric, layers=2
    )
    mc.kernel()

    refmc = mcscf.CASCI(mf, 2, 2)
    refmc.kernel()

    assert utils.almost_equal(mc.e_tot, refmc.e_tot)


def test_ssvqecasci_h2o_2e_2o():

    n_electron = 2
    n_orbitals = 2
    excitation_number = 5

    # Generate inital states
    inital_states = _generate_inital_states(n_electron, n_orbitals, excitation_number)
    state_2 = inital_states[2]
    state_3 = inital_states[3]

    # Define spin-adapted linear combinations
    state_2_spin_adapted = comp_basis_superposition(
        state_2, state_3, theta=+pi / 4, phi=0.0
    )
    state_3_spin_adapted = comp_basis_superposition(
        state_2, state_3, theta=-pi / 4, phi=0.0
    )

    # Replace initial states with spin-adapted states. Needed as Ansatz is spin conserving
    inital_states[2] = state_2_spin_adapted
    inital_states[3] = state_3_spin_adapted

    mc = vqemcscf.VQECASCI(
        mf,
        n_electron,
        n_orbitals,
        optimizer=LBFGS(),
        inital_states=inital_states,
        ansatz=Ansatz.GateFabric,
        layers=2,
        excitation_number=inital_states,
    )

    # Create reference CASCI values
    mc.kernel()
    refmc = mcscf.CASCI(mf, n_orbitals, n_electron)
    refmc.fcisolver.nroots = excitation_number
    refmc.kernel()

    def all_elements_within_threshold(array1, array2, threshold=10**-7):
        for elem in array1:
            if not any(abs(elem - other) <= threshold for other in array2):
                return False
        return True

    assert all_elements_within_threshold(mc.fcisolver.energies, refmc.e_tot)


# def test_vqecasci_h2o_4e_4o():
#    mc = vqemcscf.VQECASCI(mf, 4, 4)
#    mc.kernel()
#
#    refmc = mcscf.CASCI(mf, 4, 4)
#    refmc.kernel()
#
#    assert utils.almost_equal(mc.e_tot, refmc.e_tot, threshold=1.0e-06)


def test_vqecasscf_h2o_2e_2o():
    mc = vqemcscf.VQECASSCF(
        mf, 2, 2, optimizer=LBFGS(), ansatz=Ansatz.GateFabric, layers=2
    )
    mc.mc2step()

    refmc = mcscf.CASSCF(mf, 2, 2)
    refmc.mc2step()

    assert utils.almost_equal(mc.e_tot, refmc.e_tot)


test_ssvqecasci_h2o_2e_2o()
