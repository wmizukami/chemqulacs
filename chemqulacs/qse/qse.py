# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce

import numpy as np
from openfermion.ops import FermionOperator
from openfermion.utils import hermitian_conjugated
from pyscf.scf.addons import partial_cholesky_orth_


class QSE(object):
    """
    Quantum Subspace Expansion is a method for computing the eigenvalues of a Hermitian operator in the subspace spanned by the reference state and it's excited states.

    Args:
        vqeci:
            VQECI object.

    """

    def __init__(self, vqeci):
        self.vqeci = vqeci

    def gen_singles_doubles(self):
        singles = []
        doubles = []
        n_electron = self.vqeci.n_electron
        n_qubit = self.vqeci.n_qubit
        # singles
        for i in range(n_electron):
            for a in range(n_electron, n_qubit):
                singles += [FermionOperator("{}^ {}".format(a, i), 1.0)]
        # doubles
        for i in range(n_electron):
            for j in range(n_electron):
                for a in range(n_electron, n_qubit):
                    for b in range(n_electron, n_qubit):
                        doubles += [
                            FermionOperator("{}^ {} {}^ {}".format(a, i, b, j), 1.0)
                        ]
        return singles, doubles

    def gen_excitation_operators(self, types="ee", n_excitations=2):
        self.e_op = []
        if types == "ee":
            if n_excitations == 2:
                singles, doubles = self.gen_singles_doubles()
                self.e_op = singles + doubles
            elif n_excitations == 1:
                singles, doubles = self.gen_singles_doubles()
                self.e_op = singles

    def build_H(self):
        noperators = len(self.e_op)
        op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            self.vqeci.n_electron, self.vqeci.n_qubit
        )
        self.hamiltonian = np.zeros(
            (noperators + 1, noperators + 1), dtype=np.complex128
        )
        self.hamiltonian[0, 0] = self.vqeci.e
        for idx, iop in enumerate(self.e_op, 1):
            myop_i_fermi = hermitian_conjugated(iop) * self.vqeci.fermionic_hamiltonian
            myop_i_qubit = op_mapper(myop_i_fermi)
            self.hamiltonian[idx, 0] = self.vqeci.estimator(
                [myop_i_qubit], [self.vqeci.opt_states[0]]
            )[0].value
            self.hamiltonian[0, idx] = self.hamiltonian[idx, 0].conj()
            for jdx, jop in enumerate(self.e_op, 1):
                myop_ij = myop_i_fermi * jop
                myop_ij = op_mapper(myop_ij)
                self.hamiltonian[idx, jdx] = self.vqeci.estimator(
                    [myop_ij], [self.vqeci.opt_states[0]]
                )[0].value
                self.hamiltonian[jdx, idx] = (self.hamiltonian[idx, jdx]).conj()

    def build_S(self):
        noperators = len(self.e_op)
        op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            self.vqeci.n_electron, self.vqeci.n_qubit
        )
        self.S = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        self.S[0, 0] = 1.0
        for idx, iop in enumerate(self.e_op, 1):
            myop_i_fermi = hermitian_conjugated(iop)
            myop_i_qubit = op_mapper(myop_i_fermi)
            self.S[idx, 0] = self.vqeci.estimator(
                [myop_i_qubit], [self.vqeci.opt_states[0]]
            )[0].value
            self.S[0, idx] = (self.S[idx, 0]).conj()
            for jdx, jop in enumerate(self.e_op, 1):
                myop_ij = myop_i_fermi * jop
                myop_ij = op_mapper(myop_ij)
                self.S[idx, jdx] = self.vqeci.estimator(
                    [myop_ij], [self.vqeci.opt_states[0]]
                )[0].value
                self.S[jdx, idx] = (self.S[idx, jdx]).conj()

    def solve(self):
        self.build_H()
        self.build_S()
        # Use PySCF's partial cholesky orthogonalization to remove linear dependency
        threshold = 1.0e-8
        cholesky_threshold = 1.0e-08

        def eigh(h, s):
            x = partial_cholesky_orth_(s, canthr=threshold, cholthr=cholesky_threshold)
            xhx = reduce(np.dot, (x.T.conj(), h, x))
            e, c = np.linalg.eigh(xhx)
            c = np.dot(x, c)
            return e, c

        self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian, self.S)
        # print up to 10 excitations
        eigenvalues = self.eigenvalues[:10]
        for idx, ie in enumerate(eigenvalues):
            print("{} {}".format(idx, ie))
