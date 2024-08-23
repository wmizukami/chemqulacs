# Chemqulacs


Chemqulacs includes quantum chemistry code running on quantum circuit simulators and quantum computers.

## Documentation

<https://wmizukami.github.io/chemqulacs/>

## Installation

Chemqulacs requires Python 3.9.8 or later.

```
pip install git+https://github.com/wmizukami/chemqulacs.git
```

## Usage

### VQECASCI, VQECASSCF and VQECI

You can perform CASCI and CASSCF with VQE by `VQECASCI` and `VQECASSCF` classes, which are implemented using PySCF.

```python
from pyscf import gto, scf
from chemqulacs.util import utils
from chemqulacs.vqe.vqemcscf import VQECASCI, VQECASSCF

# Retrieve geometry of a water molecule from PubChem
geom_water = utils.get_geometry_from_pubchem("water")
# Create PySCF's Mole object
mol = gto.M(atom=geom_water, basis="sto-3g")
# Run SCF calculation (RHF)
mf = scf.RHF(mol)
mf.run()

# Run CASCI with VQE with CAS(2e, 2o)
vqe_casci = VQECASCI(mf, 2, 2)
vqe_casci.kernel()
print(f"VQE-CASCI Energy: {vqe_casci.e_tot}")

# Run CASSCF with VQE with CAS(2e, 2o)
vqe_casscf = VQECASSCF(mf, 2, 2)
vqe_casscf.kernel()
print(f"VQE-CASSCF Energy: {vqe_casscf.e_tot}")
```

The default options for VQE are as follows:

- Fermion-qubit mapping: Jordan-Wigner
- Optimizer: Adam
- Quantum backend: Qulacs
- Ansatz: ParticleConservingU1 with 2 layers

You can change those options via arguments for `VQECASCI` and `VQECASSCF` objects, which are then passed to `VQECI` objects. Please see [`chemqulacs.vqe.vqeci` module](https://wmizukami.github.io/chemqulacs/chemqulacs.vqe.vqeci.html) for more details.


#### with SSVQE
You can perform CASCI and CASSCF with SSVQE
```python
from pyscf import gto, scf
from chemqulacs.util import utils
from chemqulacs.vqe.vqemcscf import VQECASCI, VQECASSCF

# Retrieve geometry of a water molecule from PubChem
geom_water = utils.get_geometry_from_pubchem("water")
# Create PySCF's Mole object
mol = gto.M(atom=geom_water, basis="sto-3g")
# Run SCF calculation (RHF)
mf = scf.RHF(mol)
mf.run()

# Run CASCI with VQE with CAS(2e, 2o)
vqe_casci = VQECASCI(mf, 2, 2,excitation_number=3)
vqe_casci.kernel()
vqe_casci.print_energies()

# Run CASSCF with VQE with CAS(2e, 2o)
vqe_casscf = VQECASSCF(mf, 2, 2,excitation_number=3)
vqe_casscf.kernel()
vqe_casscf.print_energies()
```

### QSE

You can run QSE by with VQE as follows.

```python
from pyscf import gto, scf
from chemqulacs.util import utils
from chemqulacs.qse.qse import QSE
from chemqulacs.vqe.vqemcscf import VQECASCI, VQECASSCF

# Retrieve geometry of a water molecule from PubChem
geom_water = utils.get_geometry_from_pubchem("water")
# Create PySCF's Mole object
mol = gto.M(atom=geom_water, basis="sto-3g")
# Run SCF calculation (RHF)
mf = scf.RHF(mol)
mf.run()

# Run CASCI with VQE with CAS(2e, 2o)
vqe_casci = VQECASCI(mf, 2, 2)
vqe_casci.kernel()
print(f"VQE-CASCI Energy: {vqe_casci.e_tot}")

# Run QSE with VQE
qse = QSE(vqe_casci.fcisolver)
qse.gen_excitation_operators("ee", 1)
qse.solve()
```


## Development

[Poetry](https://python-poetry.org/docs/cli/) is used for dependency management.

```
git clone https://github.com/wmizukami/chemqulacs
cd chemqulacs
poetry install
poetry run pytest
```

See `Makefile` for other commands for development.
