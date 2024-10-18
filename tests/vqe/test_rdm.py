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
from quri_parts.core.state import ComputationalBasisState
from quri_parts.openfermion.transforms import (
    bravyi_kitaev,
    jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
)
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
)

from chemqulacs.vqe.rdm import get_1rdm, get_2rdm


def test_get_1rdm():
    n_sorbs = 4
    n_electrons = 2
    occupied_indices = [0, 1]
    estimator = create_qulacs_vector_concurrent_estimator()

    jw_mapping = jordan_wigner
    state_mapper = jw_mapping.get_state_mapper(n_sorbs, n_electrons, 0)
    jw_state = state_mapper(occupied_indices)
    jw_1rdm = get_1rdm(jw_state, jw_mapping, estimator, n_electrons)
    assert np.allclose(
        jw_1rdm,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    )

    bk_mapping = bravyi_kitaev
    state_mapper = bk_mapping.get_state_mapper(n_sorbs, n_electrons, 0)
    bk_state = state_mapper(occupied_indices)
    assert np.allclose(
        jw_1rdm,
        get_1rdm(bk_state, bk_mapping, estimator, n_electrons),
    )

    scbk_mapping = symmetry_conserving_bravyi_kitaev
    state_mapper = scbk_mapping.get_state_mapper(n_sorbs, n_electrons, 0)
    scbk_state = state_mapper(occupied_indices)
    assert np.allclose(
        jw_1rdm,
        get_1rdm(scbk_state, scbk_mapping, estimator, n_electrons),
    )


def test_get_2rdm():
    n_sorbs = 4
    n_electrons = 2
    bits = 0
    for i in range(n_electrons):
        bits += 1 << i
    state = ComputationalBasisState(n_sorbs, bits=bits)
    estimator = create_qulacs_vector_concurrent_estimator()

    jw_mapping = jordan_wigner
    jw_2rdm = get_2rdm(state, jw_mapping, estimator, n_electrons)

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
    assert np.allclose(jw_2rdm, expected)

    bk_mapping = bravyi_kitaev
    state_mapper = bk_mapping.get_state_mapper(n_sorbs, n_electrons, 0)
    occupied_indices = [0, 1]
    bk_state = state_mapper(occupied_indices)
    assert np.allclose(jw_2rdm, get_2rdm(bk_state, bk_mapping, estimator, n_electrons))

    scbk_mapping = symmetry_conserving_bravyi_kitaev
    state_mapper = scbk_mapping.get_state_mapper(n_sorbs, n_electrons, 0)
    occupied_indices = [0, 1]
    scbk_state = state_mapper(occupied_indices)
    assert np.allclose(
        jw_2rdm, get_2rdm(scbk_state, scbk_mapping, estimator, n_electrons)
    )
