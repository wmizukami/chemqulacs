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

from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf
from chemqulacs.vqe.vqeci import Ansatz

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
    mc = vqemcscf.VQECASCI(
        mf,
        2,
        2,
        optimizer=LBFGS(),
        ansatz=Ansatz.GateFabric,
        layers=2,
    )
    mc.fcisolver.nroots = 4
    mc.kernel()

    # refmc = mcscf.CASCI(mf, 4, 4)
    # refmc.fcisolver.nroots = 5
    # refmc.kernel()
    # print(refmc.e_tot)

    ref_energies = [
        -74.96569511044997,
        -74.56710399202521,
        -74.52886408322473,
        -74.52886408322476,
    ]
    assert all(
        utils.almost_equal(a, b) for a, b in zip(mc.fcisolver.energies, ref_energies)
    )


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
