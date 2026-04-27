##################
# Library imports #
###################

import sys

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, dump

from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks import site
from tenpy.algorithms import tdvp


class Ising_Model(CouplingMPOModel):
    
    def init_sites(self, model_params):
        return site.SpinHalfSite(conserve=None)
    
    # Define model
    def init_terms(self, model_params):
        
        J = model_params.get('J', 0.5)
        g = model_params.get('g', 1.0)


        # Add coupling for all nearest neighbor pairs
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(1.0, u1, 'Sigmaz', u2, 'Sigmaz', dx)

        # Add next-nearest neighbor coupling

        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(J, u1, 'Sigmaz', u2, 'Sigmaz', dx)

        self.add_onsite(g, 0, 'Sigmax')

def random_state(L, chi_max, sites):

    psi_random = np.random.choice(['up', 'down'], size=L)

    MPS_random = MPS.from_random_unitary_evolution(sites, chi_max, psi_random, bc='finite', dtype=float)

    Sx_avg = np.mean(MPS_random.expectation_value('Sigmax'))
    Sy_avg = np.mean(MPS_random.expectation_value('Sigmay'))
    Sz_avg = np.mean(MPS_random.expectation_value('Sigmaz'))

    return MPS_random, Sx_avg, Sy_avg, Sz_avg

def spectral_function(psi0, model, times, tdvp_params):

    L = psi0.L
    S_t = np.zeros(len(times), dtype=complex)

    # Computes the first step in the time evolution of the spectral function

    #S_t[0] = psi0.overlap(psi0) # S(0) = ⟨ψ(0)|ψ(0)⟩

    # Copy do avoid overwriting original state

    ket_0 = psi0.copy()
    bra_0 = psi0.copy()

    ket_0.apply_local_op(L//2, 'Sz') # Sz|ψ(0)⟩

    ket_eng = tdvp.TwoSiteTDVPEngine(ket_0, model, tdvp_params)
    bra_eng = tdvp.TwoSiteTDVPEngine(bra_0, model, tdvp_params)

    for ti in range(1, len(times)):

        ket_eng.run()
        bra_eng.run()

        ket = ket_eng.psi # e^(-iHt) Sz|ψ(0)⟩
        bra = bra_eng.psi # e^(-iHt)|ψ(0)⟩

        psi = ket.copy()
        phi = bra.copy()

        phi.apply_local_op(L//2, 'Sz') # S_z e^(-iHt)|ψ(0)⟩

        S_t[ti] = phi.overlap(psi) # S(t) = ⟨ψ(t)|φ(t)⟩ = ⟨ψ(0)|e^(iHt) S_z e^(-iHt) S_z|ψ(0)⟩

    return S_t


######################################################################################
# Constructs an ensemble of random states and builds respective MPS representations. #
######################################################################################

L = int(sys.argv[1])
J = float(sys.argv[2])
g = float(sys.argv[3])

dt = 0.1
t_max = float(sys.argv[4])
times = np.linspace(0, t_max, int(t_max/dt))

N_ensemble = int(sys.argv[5])
chi_max = 4

task_id = int(sys.argv[6])

#################################
# Bookkeeping: print statements #
#################################

model_params = {
    'lattice': 'Chain', 
    'L': L,
    'J': J,
    'g': g,
    'bc_MPS': 'finite',
}

tdvp_params = {
   'N_steps': 1,
    'dt': dt,
    'trunc_params': {
        'chi_max': 100, 
        'svd_min': 1.e-6,
    },
}

model = Ising_Model(model_params)

sites = model.lat.mps_sites()

results = Parallel(n_jobs=-1)(delayed(random_state)(L, chi_max, sites) for _ in range(N_ensemble))

ensemble, Sx, Sy, Sz = zip(*results)

norms = np.sqrt(np.array([psi.overlap(psi) for psi in ensemble]))

print()

print(f'Model parameters: g={g}, J={J}')
print(f'Simulation parameters: L={L}, tmax={t_max}, Ensemble size={N_ensemble}')

print()

print('Ensemble construction completed!')

print(f'Average norm across ensemble: {np.mean(norms):.4f} ± {np.std(norms):.4f}')

print(f'Average Sx across ensemble: {np.mean(Sx):.4f} ± {np.std(Sx):.4f}')
print(f'Average Sy across ensemble: {np.mean(Sy):.4f} ± {np.std(Sy):.4f}')
print(f'Average Sz across ensemble: {np.mean(Sz):.4f} ± {np.std(Sz):.4f}')

print('=' * 100)
print()

#####################################################################################
#         Real-time evolution to compute spectral function S(t) = ⟨ψ(t)|φ(t)⟩        #
#         where φ(t) = e^(-iHt) Sz|ψ(0)⟩ and ψ(t) = Sz e^(-iHt)|ψ(0)⟩                #
#####################################################################################

# Run spectral function calculation in parallal for all states in the ensemble

S_t = Parallel(n_jobs=-1, verbose=10)(delayed(spectral_function)(psi, model, times, tdvp_params) for psi in ensemble)

S_t = np.array(S_t)

filename_S = f"S_L{L}_tmax{t_max}_g{g}_J{J}_id{task_id}.npy"
np.save(filename_S, S_t)

print('Simulation completed!')



