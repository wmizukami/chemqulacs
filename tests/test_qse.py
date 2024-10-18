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
from qulacs import QuantumState

from chemqulacs.qse import qse
from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf

n_qubit = 8
n_electron = 4
state = QuantumState(n_qubit)
state.set_computational_basis(
    int("0b" + "0" * (n_qubit - n_electron) + "1" * (n_electron), 2)
)  # Hartree-Fock State

geom_water = utils.get_geometry_from_pubchem("water")
mol = gto.M(atom=geom_water, basis="sto-3g")

# SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
mf = scf.RHF(mol)
# SCF計算の実行, エネルギーが得られる(-74.96444758277)
mf.run()

# CASCIのオブジェクトを生成。ここではRHF波動関数を読み込ませて、その分子軌道を使用。
# 活性軌道にはCAS(2e,2o)を使用する
refmc = mcscf.CASCI(mf, 2, 2)
refmc.run()

# 量子古典混合アルゴリズムであるVQEを使ったCASCIを実行する
# （より正確にはactive space disentangled unitary coupled cluster)
# 活性軌道にはCAS(2e,2o)を使用する
mc = vqemcscf.VQECASCI(mf, 2, 2)
mc.kernel()


def test_qse_init():
    qse.QSE(mc.fcisolver)


def test_qse_gen_singles_and_doubles():
    qse.QSE(mc.fcisolver).gen_singles_doubles()


def test_qse_solve_singles():
    myqse = qse.QSE(mc.fcisolver)
    myqse.gen_excitation_operators("ee", 1)
    myqse.solve()


# def test_qse_solve_doubles():
#    myqse = qse.QSE(mc.fcisolver)
#    myqse.gen_excitation_operators("ee", 2)
#    myqse.solve()


# def test_qse_init():
if __name__ == "__main__":
    # from pyscf import cc
    # mycc = cc.CCSD(mf).run()
    # e_ee, c_ee = mycc.eeccsd(nroots=5)
    # print (e_ee)
    # e_s, c_s = mycc.eomee_ccsd_singlet(nroots=5)
    # sprint (e_s)
    myqse = qse.QSE(mc.fcisolver)
    # singles, doubles = myqse.gen_singles_doubles()
    # print (singles)
    # print (doubles)
    myqse.gen_excitation_operators("ee", 1)
    # print (myqse.e_op)
    # myqse.build_H()
    # myqse.build_S()
    # print (myqse.hamiltonian)
    myqse.solve()
