# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from chemqulacs.util import utils


def test_almost_equal():
    assert utils.almost_equal(1.000000001, 1.000000000)


def test_get_geometry_from_pubchem():
    geom_water = utils.get_geometry_from_pubchem("water")
    from pyscf import gto, scf

    mol = gto.M(atom=geom_water, basis="sto-3g")
    mf = scf.RHF(mol).density_fit()
    mf.run()
    energy = mf.energy_tot()
    assert utils.almost_equal(energy, -74.9645350309677)
