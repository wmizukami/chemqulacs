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
vqe_casci = VQECASCI(mf, 2, 2,excitation_number=3)
vqe_casci.kernel()
print(f"VQE-CASCI Energy: {vqe_casci.e_tot}")

# Run QSE with VQE
qse = QSE(vqe_casci.fcisolver)
qse.gen_excitation_operators("ee", 1)
qse.solve()