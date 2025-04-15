#%%
import qutip as qt
import numpy as np

#%%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- System Parameters ---
g = 1.0  # coupling strength
t_swap = np.pi / (2 * g)  # full excitation transfer

H = g * (
    tensor(destroy(2), sigmap())
    + tensor(create(2), sigmam())
)

H

#%%
tensor(basis(2, 1), basis(2, 0))

#%%
U = (-1j * H * t_swap).expm()  # exact unitary evolution
U


#%%
# H = g * (
#     tensor(destroy(2), identity(2), sigmap(), identity(2), identity(2))
#     + tensor(create(2), identity(2), sigmam(), identity(2), identity(2))
#     + tensor(identity(2),  destroy(2), identity(2), sigmap(), identity(2))
#     + tensor(identity(2),  create(2), identity(2), sigmam(), identity(2))
# )
H = g * (
    tensor(destroy(2), identity(2), sigmap(), identity(2))
    + tensor(create(2), identity(2), sigmam(), identity(2))
    + tensor(identity(2),  destroy(2), identity(2), sigmap())
    + tensor(identity(2),  create(2), identity(2), sigmam())
)

H
#%%
# --- Initial State ---
phi = np.pi * 0.1 # photon phase to transduce

# Photon in (|0⟩ + e^{iφ}|1⟩)/√2, qubit in |g⟩
psi_photon = (
    tensor(basis(2, 0), basis(2, 1)) 
    + np.exp(1j * phi) *  tensor(basis(2, 1), basis(2, 0))
).unit()

# psi_qubit = (
#     tensor(basis(2, 0), basis(2, 0))
#     + tensor(basis(2, 0), basis(2, 1))
#     + tensor(basis(2, 1), basis(2, 0))
#     - tensor(basis(2, 1), basis(2, 1))
# ).unit()
psi_qubit = tensor(basis(2, 1), basis(2, 0)) #, basis(2, 0)) # |g⟩
psi_photon
psi_qubit

#%%
psi0 = tensor(psi_photon, psi_qubit)

# --- Time Evolution ---
U = (-1j * H * t_swap).expm()  # exact unitary evolution
psi_final = U * psi0
print("Final state:\n", psi_final)
rho_final = psi_final.proj()

#%%
# --- Extract qubit reduced state ---
rho_qubit = rho_final.ptrace([2, 3])  # trace out photon
# rho_qubit = rho_final.ptrace([2, 3, 4])  # trace out photon
# print("\nReduced qubit state:\n", rho_qubit)
rho_qubit

#%%
rho_photon = rho_final.ptrace([0, 1])  # trace out qubit
# print("Reduced photon state:\n", rho_photon)
rho_photon 

#%%
print(f"Photon purity {rho_photon.purity()}")
print(f"Qubit purity {rho_qubit.purity()}")

#%%
# --- Visualize Bloch vector ---
# b = Bloch()
# b.add_states(rho_qubit)
# b.make_sphere()
# b.show()