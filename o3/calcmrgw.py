from pyscf import gto, scf, mcscf, symm
from gf.gw.mrgw import CASGW
from gf.gw.addons import wrt_gf_dat, read_spec_dat, anlz_gf_residue
from scipy.signal import find_peaks
import numpy as np
from gf.addons import printci
from pyscf.data.nist import HARTREE2EV as au2ev
import cupy as cp

mol = gto.M()

r = 1.278
#theta = 175.

eta = .1/au2ev*.5

#for theta in range(90,150,4):
#for theta in range(146,147,4):
for theta in [92]:

    print('Theta = ', theta)
    
    s = theta/180*np.pi/2

    mol.atom = [['O', (0., 0., 0.)],
    ['O', (0., r*np.sin(s), r*np.cos(s))],
    ['O', (0., -r*np.sin(s), r*np.cos(s))]]

    mol.basis = '6-31G'
    mol.spin = 0
    mol.symmetry = 1
    mol.max_memory = 40000
    mol.verbose = 4
    mol.build()

    rhf = scf.RHF(mol)#.newton()
    rhf.kernel()

    mo,_,stable,_ = rhf.stability(return_status=True)

    while (not stable):
        dm = rhf.make_rdm1(mo,rhf.mo_occ)
        rhf.kernel(dm)
        mo,_,stable,_ = rhf.stability(return_status=True)

    _,mo,_,stable = rhf.stability(return_status=True, external=True)

    rhf.analyze()

    orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, rhf.mo_coeff)


    ncas = {'A2':1, 'B1':2, 'A1':2, 'B2':1}
    ncore = {'A1':5, 'B1':0, 'B2':3, 'A2':0}

    #ncas = {'A2':1, 'B1':2, 'A1':1, 'B2':1}
    cas = mcscf.CASCI(rhf, ncas=6, nelecas=(4,4))

    #ncas = {'A2':1, 'B1':2, 'A1':3, 'B2':3}
    #cas = mcscf.CASCI(rhf, ncas=9, nelecas=(6,6))

    cas.verbose = 4
    mo_coeff = mcscf.sort_mo_by_irrep(cas, rhf.mo_coeff, ncas, cas_irrep_ncore=ncore)
    cas.kernel(mo_coeff)

    printci(cas.ci, cas.ncas, cas.nelecas)

    casgw = CASGW(cas, ss=0.)
    Gcas,Gcasgw = casgw.G(eta=eta, batchsize=5000)

    w0 = -30./au2ev
    w1 = -5./au2ev

    step = .2*eta

    print('---- Write Gcas ----')

    fnam = f'spec-o3-Gcas86-{theta:03d}.dat'
    wrt_gf_dat(Gcas, w0, w1, step=step, fnam=fnam)
    freq, spec = read_spec_dat(fnam)
    pk, _ = find_peaks(spec, height=.1)
    poles = (freq[pk])[-3:]

    for p in poles[::-1]:
        sp = anlz_gf_residue(Gcas, p)
        iorb = sp.argmax()%cas.ncas+cas.ncore
    
        print('IP(CAS) %s =%16.10f'%(orbsym[iorb], -p*au2ev))

    print('---- Write Gcasgw ----')
    fnam = f'spec-o3-Gcas86gw-{theta:03d}.dat'
    wrt_gf_dat(Gcasgw, w0, w1, step=step, fnam=fnam)
    freq, spec = read_spec_dat(fnam)
    pk, _ = find_peaks(spec, height=.1)
    poles = (freq[pk])[-3:]

    for p in poles[::-1]:
        sp = anlz_gf_residue(Gcasgw, p, r=.05/au2ev)
        iorb = sp.argmax()%cas.ncas+cas.ncore
    
        print('IP(CASGW) %s =%16.10f'%(orbsym[iorb], -p*au2ev))

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
