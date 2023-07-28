# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf.mcscf import casci, mc1step
from quri_parts.algo.optimizer import Adam
from quri_parts.openfermion.transforms import jordan_wigner

from chemqulacs.vqe.vqeci import VQECI, Ansatz, Backend, QulacsBackend


class VQECASCI(casci.CASCI):
    """
    VQECASCI
        Args:
            mf:
                SCF or Mole to define the problem size.
            ncas (int):
                Number of active spatial orbitals.
            nelecas (int):
                Number of active electrons.
            fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping):
                Mapping from :class:`FermionOperator` or :class:`InteractionOperator` to :class:`Operator`
            optimizer
            backend (Backend)
            shots_per_iter (int)
            ansatz: ansatz used for VQE
            layers (int):
                Layers of gate operations. Used for ``HardwareEfficient``, ``SymmetryPreserving``, ``ParticleConservingU1``, ``ParticleConservingU2``, and ``GateFabric``.
            k (int):
                Number of repetitions of excitation gates. Used for ``KUpCCGSD``.
            trotter_number (int):
                Number of trotter decomposition. Used for ``UCCSD`` and ``kUpCCGSD``.
            include_pi (bool):
                If ``True``, the optional constant gate is inserted. Used for ``GateFabric``.
            use_singles: (bool):
                If ``True``, single-excitation gates are applied. Used for ``UCCSD``.
            delta_sz (int):
                Changes of spin in the excitation. Used for ``KUpCCGSD``.
            singlet_excitation (bool):
                If ``True``, the ansatz will be spin symmetric. Used for ``UCCSD`` and
                ``KUpCCGSD``.
            is_init_random (bool):
                If ``False``, initial parameters are initialized to 0s, else, initialized randomly.
            seeed (int):
                Random seed.
        Return:
            None
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        fermion_qubit_mapping=jordan_wigner,
        optimizer=Adam(),
        backend: Backend = QulacsBackend(),
        shots_per_iter: int = 10000,
        ansatz: Ansatz = Ansatz.ParticleConservingU1,
        layers: int = 1,
        k: int = 1,
        trotter_number: int = 1,
        include_pi: bool = False,
        use_singles: bool = True,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
        is_init_random: bool = False,
        seed: int = 0,
    ):
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.fcisolver = VQECI(
            mf.mol,
            fermion_qubit_mapping=fermion_qubit_mapping,
            optimizer=optimizer,
            backend=backend,
            shots_per_iter=shots_per_iter,
            ansatz=ansatz,
            layers=layers,
            k=k,
            trotter_number=trotter_number,
            include_pi=include_pi,
            use_singles=use_singles,
            delta_sz=delta_sz,
            singlet_excitation=singlet_excitation,
            is_init_random=is_init_random,
            seed=seed,
        )


class VQECASSCF(mc1step.CASSCF):
    """
    VQECASSCF
        Args:
            mf:
                SCF or Mole to define the problem size.
            ncas (int):
                Number of active spatial orbitals.
            nelecas (int):
                Number of active electrons.
            fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping):
                Mapping from :class:`FermionOperator` or :class:`InteractionOperator` to :class:`Operator`
            optimizer
            backend (Backend)
            shots_per_iter (int)
            ansatz: ansatz used for VQE
            layers (int):
                Layers of gate operations. Used for ``HardwareEfficient``, ``SymmetryPreserving``, ``ParticleConservingU1``, ``ParticleConservingU2``, and ``GateFabric``.
            k (int):
                Number of repetitions of excitation gates. Used for ``KUpCCGSD``.
            trotter_number (int):
                Number of trotter decomposition. Used for ``UCCSD`` and ``kUpCCGSD``.
            include_pi (bool):
                If ``True``, the optional constant gate is inserted. Used for ``GateFabric``.
            use_singles: (bool):
                If ``True``, single-excitation gates are applied. Used for ``UCCSD``.
            delta_sz (int):
                Changes of spin in the excitation. Used for ``KUpCCGSD``.
            singlet_excitation (bool):
                If ``True``, the ansatz will be spin symmetric. Used for ``UCCSD`` and
                ``KUpCCGSD``.
            is_init_random (bool):
                If ``False``, initial parameters are initialized to 0s, else, initialized randomly.
            seeed (int):
                Random seed.
        Return:
            None
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        nfrozen=None,
        fermion_qubit_mapping=jordan_wigner,
        optimizer=Adam(),
        backend: Backend = QulacsBackend(),
        shots_per_iter: int = 10000,
        ansatz: Ansatz = Ansatz.ParticleConservingU1,
        layers: int = 1,
        k: int = 1,
        trotter_number: int = 1,
        include_pi: bool = False,
        use_singles: bool = True,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
        is_init_random: bool = False,
        seed: int = 0,
    ):
        mc1step.CASSCF.__init__(self, mf, ncas, nelecas, ncore, nfrozen)
        self.fcisolver = VQECI(
            mf.mol,
            fermion_qubit_mapping=fermion_qubit_mapping,
            optimizer=optimizer,
            backend=backend,
            shots_per_iter=shots_per_iter,
            ansatz=ansatz,
            layers=layers,
            k=k,
            trotter_number=trotter_number,
            include_pi=include_pi,
            use_singles=use_singles,
            delta_sz=delta_sz,
            singlet_excitation=singlet_excitation,
            is_init_random=is_init_random,
            seed=seed,
        )
