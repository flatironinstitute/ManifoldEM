from os import remove
from shutil import rmtree, copytree, copy

from ManifoldEM import p

'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)
'''

def op(PrD,choice):
    for j in range(p.num_psis):
        # dirs with frames
        subdir = p.out_dir+'/topos/PrD_{}/psi_{}'.format(PrD,j+1)
        subdir1 = p.out_dir + '/topos/PrD_{}/psi_orig{}'.format(PrD,j+1)
        # movies
        mov_file = p.out_dir + '/topos/PrD_{}/psi_{}.gif'.format(PrD, j + 1)
        mov_orig_file = p.out_dir+'/topos/PrD_{}/psi_{}_orig.gif'.format(PrD,j+1)
        # topos
        tm_file = '{}/topos/PrD_{}/topos_{}.png'.format(p.out_dir, PrD, j + 1)
        tm_orig_file = '{}/topos/PrD_{}/topos_orig_{}.png'.format(p.out_dir, PrD, j + 1)
        if choice == 1:  # need to make backup copy
            # dirs with frames
            rmtree(subdir1)
            copytree(subdir, subdir1)
            # movies
            copy(mov_file, mov_orig_file)
            # topos
            copy(tm_file, tm_orig_file)
        else:  # restore original backup copy
            # dirs with frames
            rmtree(subdir)
            copytree(subdir1, subdir)
            rmtree(subdir1)
            # movies
            copy(mov_orig_file, mov_file)
            remove(mov_orig_file)
            # topos
            copy(tm_orig_file, tm_file)
            remove(tm_orig_file)

    # diff maps
    psi_file = '{}prD_{}'.format(p.psi_file, PrD-1)
    psi_orig_file = '{}orig_prD_{}'.format(p.psi_file, PrD- 1)
    # psianalysis
    psi2_file = '{}prD_{}'.format(p.psi2_file, PrD - 1)
    psi2_orig_file = '{}orig_prD_{}'.format(p.psi2_file, PrD - 1)
    # class avg
    ca_file = '{}/topos/PrD_{}/class_avg.png'.format(p.out_dir, PrD)
    ca_orig_file = '{}/topos/PrD_{}/class_avg_orig.png'.format(p.out_dir, PrD)
    # eig spectrum
    eig_file = '{}/topos/PrD_{}/eig_spec.txt'.format(p.out_dir, PrD)
    eig_orig_file = '{}/topos/PrD_{}/eig_spec_orig.txt'.format(p.out_dir, PrD)


    if choice == 1:
        copy(psi_file, psi_orig_file)
        copy(ca_file, ca_orig_file)
        copy(eig_file, eig_orig_file)

        for i in range(p.num_psis):
            psi2 = '{}_psi_{}'.format(psi2_file, i)
            psi2_orig = '{}_psi_{}'.format(psi2_orig_file, i)
            copy(psi2, psi2_orig)

    else:
        copy(psi_orig_file, psi_file)
        remove(psi_orig_file)
        copy(ca_orig_file, ca_file)
        remove(ca_orig_file)
        copy(eig_orig_file, eig_file)
        remove(eig_orig_file)

        for i in range(p.num_psis):
            psi2 = '{}_psi_{}'.format(psi2_file, i)
            psi2_orig = '{}_psi_{}'.format(psi2_orig_file, i)
            copy(psi2_orig, psi2)
            remove(psi2_orig)

