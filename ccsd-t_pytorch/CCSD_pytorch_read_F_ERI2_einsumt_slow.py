"""
Script to compute the electronic correlation energy using
coupled-cluster theory through single and double excitations,
from a RHF reference wavefunction.
References:
- Algorithms from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
- DPD Formulation of CC Equations: [Stanton:1991:4334]
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith", "Lori A. Burns"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-07-29"

import time
import numpy as np
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import psi4
import sys
from einsumt import einsumt as einsum_mt

'''
# Set memory
psi4.set_memory('32 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 32

mol = psi4.geometry("""
  c -0.761877 -0.000000 -0.000000
  c 0.761877 -0.000000 -0.000000
  h -1.154063 0.565714 -0.836802
  h -1.154065 -1.007548 -0.071517
  h -1.154063 0.441840 0.908322
  h 1.154064 -0.565709 0.836806
  h 1.154059 -0.441842 -0.908321
  h 1.154069 1.007548 0.071515

symmetry c1
""")
'''
#psi4.set_options({'basis': 'cc-pvdz',
#                  'scf_type': 'df',
#                  'mp2_type': 'conv',
#                  'freeze_core': 'false',
#                  'e_convergence': 1e-10,
#                  'd_convergence': 1e-10})

# CCSD Settings
E_conv = 1.e-6
maxiter = 50
#print_amps = False
#compare_psi4 = False

# First compute RHF energy using Psi4
#scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from
#C = wfn.Ca()
#ndocc = wfn.doccpi()[0]
#nmo = wfn.nmo()
#SCF_E = wfn.energy()
#eps = np.asarray(wfn.epsilon_a())

nmo = int(sys.argv[1])
ndocc = int(sys.argv[2])

# Compute size of SO-ERI tensor in GB
#ERI_Size = (nmo ** 4) * 128e-9
#print('\nSize of the SO ERI tensor will be %4.2f GB.' % ERI_Size)
#memory_footprint = ERI_Size * 5.2
#if memory_footprint > numpy_memory:
#    clean()
#    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
#                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
#t = time.time()
#mints = psi4.core.MintsHelper(wfn.basisset())
#H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

#print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

# Make spin-orbital MO antisymmetrized integrals
#print('Starting AO -> spin-orbital MO transformation...')
#MO = np.asarray(mints.mo_spin_eri(C, C))
# read from FCIDUMP
eri = np.load('ERI.npy')
MO = eri - np.swapaxes(eri, 2, 3)
del eri
eri2 = np.load('ERI2.npy')
MO2 = eri2 - np.swapaxes(eri2, 2, 3)
del eri2

# Update nocc and nvirt
nso = nmo * 2
nocc = ndocc * 2
nvirt = nso - nocc

# Make slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])


#Extend eigenvalues
#eps = np.repeat(eps, 2)
#Eocc = eps[o]
#Evirt = eps[v]

#print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

# DPD approach to CCSD equations from [Stanton:1991:4334]

# occ orbitals i, j, k, l, m, n
# virt orbitals a, b, c, d, e, f
# all oribitals p, q, r, s, t, u, v


#Bulid Eqn 9: tilde{\Tau})
def build_tilde_tau(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 9"""
    ttau = t2.copy()
    tmp = 0.5 * einsum_mt('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    del tmp
    return ttau


#Build Eqn 10: \Tau)
def build_tau(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 10"""
    ttau = t2.copy()
    tmp = einsum_mt('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    del tmp
    return ttau


#Build Eqn 3:
def build_Fae(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 3"""
    Fae = F[v, v].copy()
    Fae[np.diag_indices_from(Fae)] = 0

    Fae -= 0.5 * einsum_mt('me,ma->ae', F[o, v], t1)
    Fae += einsum_mt('mf,mafe->ae', t1, MO[o, v, v, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fae -= 0.5 * einsum_mt('mnaf,mnef->ae', tmp_tau, MO[o, o, v, v])
    del tmp_tau
    return Fae


#Build Eqn 4:
def build_Fmi(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 4"""
    Fmi = F[o, o].copy()
    Fmi[np.diag_indices_from(Fmi)] = 0

    Fmi += 0.5 * einsum_mt('ie,me->mi', t1, F[o, v])
    Fmi += einsum_mt('ne,mnie->mi', t1, MO[o, o, o, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fmi += 0.5 * einsum_mt('inef,mnef->mi', tmp_tau, MO[o, o, v, v])

    del tmp_tau
    return Fmi


#Build Eqn 5:
def build_Fme(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 5"""
    Fme = F[o, v].copy()
    Fme += einsum_mt('nf,mnef->me', t1, MO[o, o, v, v])
    return Fme


#Build Eqn 6:
def build_Wmnij(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 6"""
    Wmnij = MO[o, o, o, o].copy()

    Pij = einsum_mt('je,mnie->mnij', t1, MO[o, o, o, v])
    Wmnij += Pij
    Wmnij -= Pij.swapaxes(2, 3)

    del Pij

    tmp_tau = build_tau(t1, t2)
    Wmnij += 0.25 * einsum_mt('ijef,mnef->mnij', tmp_tau, MO[o, o, v, v])
    
    del tmp_tau 
    return Wmnij


#Build Eqn 7:
def build_Wabef(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 7"""
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper

    Wabef = MO[v, v, v, v].copy()

    Pab = einsum_mt('baef->abef', np.tensordot(t1, MO[v, o, v, v], axes=(0, 1)))
    # Pab = einsum_mt('mb,amef->abef', t1, MO[v, o, v, v])

    Wabef -= Pab
    Wabef += Pab.swapaxes(0, 1)

    del Pab

    tmp_tau = build_tau(t1, t2)

    Wabef += 0.25 * np.tensordot(tmp_tau, MO[v, v, o, o], axes=((0, 1), (2, 3)))
    # Wabef += 0.25 * einsum_mt('mnab,mnef->abef', tmp_tau, MO[o, o, v, v])
    del tmp_tau 
    return Wabef



#Build Eqn 8:
def build_Wmbej(t1, t2):
    """Builds [Stanton:1991:4334] Eqn. 8"""
    Wmbej = MO[o, v, v, o].copy()
    Wmbej += einsum_mt('jf,mbef->mbej', t1, MO[o, v, v, v])
    Wmbej -= einsum_mt('nb,mnej->mbej', t1, MO[o, o, v, o])

    tmp = (0.5 * t2) + einsum_mt('jf,nb->jnfb', t1, t1)

    Wmbej -= einsum_mt('jbme->mbej', np.tensordot(tmp, MO[o, o, v, v], axes=((1, 2), (1, 3))))
    # Wmbej -= einsum_mt('jnfb,mnef->mbej', tmp, MO[o, o, v, v])

    del tmp
    return Wmbej

### Build so Fock matirx

# Update H, transform to MO basis and tile for alpha/beta spin
#H = np.einsum('uj,vi,uv', C, C, H)
#H = np.repeat(H, 2, axis=0)
#H = np.repeat(H, 2, axis=1)

# Make H block diagonal
#spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
#H *= (spin_ind.reshape(-1, 1) == spin_ind)

# Compute Fock matrix
#F = H + np.einsum('pmqm->pq', MO[:, o, :, o])
F = np.load('F.npy')

#print(F)
#print(MO.shape)

### Build D matrices: [Stanton:1991:4334] Eqns. 12 & 13
Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()

Dia = Focc.reshape(-1, 1) - Fvirt
Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt

### Construct initial guess

# t^a_i
t1 = np.zeros((nocc, nvirt))
# t^{ab}_{ij}
MOijab = MO[o, o, v, v]
MO2ijab = MO2[o, o, v, v].copy()
del MO2
t2 = MOijab / Dijab

### Compute MP2 in MO basis set to make sure the transformation was correct
MP2corr_E = np.einsum('ijab,ijab->', MOijab, t2) / 4
#MP2_E = SCF_E + MP2corr_E

print('MO based MP2 correlation energy: %.8f' % MP2corr_E)
#print('MP2 total energy:       %.8f' % MP2_E)
#psi4.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')

### Start CCSD iterations
print('\nStarting CCSD iterations')
ccsd_tstart = time.time()
CCSDcorr_E_old = 0.0
#np.save('INTDUMP_ccsd.npy', MOijab)

for CCSD_iter in range(1, maxiter + 1):
    ### Build intermediates: [Stanton:1991:4334] Eqns. 3-8
    Fae = build_Fae(t1, t2)
    Fmi = build_Fmi(t1, t2)
    Fme = build_Fme(t1, t2)

    Wmnij = build_Wmnij(t1, t2)
    Wabef = build_Wabef(t1, t2)
    Wmbej = build_Wmbej(t1, t2)

    #### Build RHS side of t1 equations, [Stanton:1991:4334] Eqn. 1
    rhs_T1  = F[o, v].copy()
    rhs_T1 += einsum_mt('ie,ae->ia', t1, Fae)
    rhs_T1 -= einsum_mt('ma,mi->ia', t1, Fmi)
    rhs_T1 += einsum_mt('imae,me->ia', t2, Fme)
    rhs_T1 -= einsum_mt('nf,naif->ia', t1, MO[o, v, o, v])
    rhs_T1 -= 0.5 * einsum_mt('imef,maef->ia', t2, MO[o, v, v, v])
    rhs_T1 -= 0.5 * einsum_mt('mnae,nmei->ia', t2, MO[o, o, v, o])

    ### Build RHS side of t2 equations, [Stanton:1991:4334] Eqn. 2
    rhs_T2 = MO[o, o, v, v].copy()

    # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
    tmp = Fae - 0.5 * einsum_mt('mb,me->be', t1, Fme)
    Pab = einsum_mt('ijae,be->ijab', t2, tmp)
    del tmp
    rhs_T2 += Pab
    rhs_T2 -= Pab.swapaxes(2, 3)
    del Pab

    # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
    tmp = Fmi + 0.5 * einsum_mt('je,me->mj', t1, Fme)
    Pij = einsum_mt('imab,mj->ijab', t2, tmp)
    del tmp
    rhs_T2 -= Pij
    rhs_T2 += Pij.swapaxes(0, 1)
    del Pij

    tmp_tau = build_tau(t1, t2)
    rhs_T2 += 0.5 * einsum_mt('mnab,mnij->ijab', tmp_tau, Wmnij)
    rhs_T2 += 0.5 * einsum_mt('ijef,abef->ijab', tmp_tau, Wabef)
    del tmp_tau
    del Wmnij
    del Wabef
    # P_(ij) * P_(ab)
    # (ij - ji) * (ab - ba)
    # ijab - ijba -jiab + jiba
    tmp = einsum_mt('ie,ma,mbej->ijab', t1, t1, MO[o, v, v, o])
    Pijab = einsum_mt('imae,mbej->ijab', t2, Wmbej)
    Pijab -= tmp
    del tmp
    del Wmbej

    rhs_T2 += Pijab
    rhs_T2 -= Pijab.swapaxes(2, 3)
    rhs_T2 -= Pijab.swapaxes(0, 1)
    rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
    del Pijab

    Pij = einsum_mt('ie,abej->ijab', t1, MO[v, v, v, o])
    rhs_T2 += Pij
    rhs_T2 -= Pij.swapaxes(0, 1)
    del Pij

    Pab = einsum_mt('ma,mbij->ijab', t1, MO[o, v, o, o])
    rhs_T2 -= Pab
    rhs_T2 += Pab.swapaxes(2, 3)
    del Pab

    ### Update t1 and t2 amplitudes
    t1 = rhs_T1 / Dia
    t2 = rhs_T2 / Dijab

    ### Compute CCSD correlation energy
    CCSDcorr_E = einsum_mt('ia,ia->', F[o, v], t1)
    CCSDcorr_E += 0.25 * einsum_mt('ijab,ijab->', MO2ijab, t2)
    CCSDcorr_E += 0.5 * einsum_mt('ijab,ia,jb->', MO2ijab, t1, t1)
    
    ### Print CCSD correlation energy
    print('CCSD Iteration %3d: CCSD correlation = %3.12f  '\
          'dE = %3.5E' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old)))
    if (abs(CCSDcorr_E - CCSDcorr_E_old) < E_conv):
        break

    CCSDcorr_E_old = CCSDcorr_E

#np.save('t1_ccsd.npy', t1)
#np.save('t2_ccsd.npy', t2)
#term = 0.25 * einsum_mt('ijab,ijab->ijab', MO[o, o, v, v], t2)
#np.save('term_spin.npy', term)

print('CCSD iterations took %.2f seconds.\n' % (time.time() - ccsd_tstart))

#CCSD_E = SCF_E + CCSDcorr_E

print('\nFinal CCSD correlation energy:     % 16.10f' % CCSDcorr_E)
#print('Total CCSD energy:                 % 16.10f' % CCSD_E)

'''
if compare_psi4:
    psi4.compare_values(psi4.energy('CCSD'), CCSD_E, 6, 'CCSD Energy')

if print_amps:
    # [::4] take every 4th, [-5:] take last 5, [::-1] reverse order
    t2_args = np.abs(t2).ravel().argsort()[::2][-5:][::-1]
    t1_args = np.abs(t1).ravel().argsort()[::4][-5:][::-1]

    print('\nLargest t1 amplitudes')
    for pos in t1_args:
        value = t1.flat[pos]
        inds = np.unravel_index(pos, t1.shape)
        print('%4d  %4d |   % 5.10f' % (inds[0], inds[1], value))

    print('\nLargest t2 amplitudes')
    for pos in t2_args:
        value = t2.flat[pos]
        inds = np.unravel_index(pos, t2.shape)
        print('%4d  %4d  %4d  %4d |   % 5.10f' % (inds[0], inds[1], inds[2], inds[3], value))
'''
