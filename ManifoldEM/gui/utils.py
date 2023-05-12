import mrcfile
import pandas
import pickle

import numpy as np

from ManifoldEM import p

def threshold_pds():
    print('thresholding PDs...')
    print('low threshold: %s' % p.PDsizeThL)
    print('high threshold: %s' % p.PDsizeThH)
    with open(p.tess_file, 'rb') as f:
        data = pickle.load(f)

    # all tessellated bins:
    totalPrDs = int(np.shape(data['CG1'])[0])
    mid = data['CG1'].shape[0] // 2
    NC1 = data['NC'][:int(mid)]
    NC2 = data['NC'][int(mid):]

    all_PrDs = []
    all_occ = []
    thresh_PrDs = []
    thresh_occ = []
    if len(NC1) >= len(NC2):  # first half of S2
        pd_all = 1
        pd = 1
        for i in NC1:
            all_PrDs.append(pd_all)
            all_occ.append(i)
            if i >= p.PDsizeThL:
                thresh_PrDs.append(pd)
                if i > p.PDsizeThH:
                    thresh_occ.append(p.PDsizeThH)
                else:
                    thresh_occ.append(i)
                pd += 1
            pd_all += 1
    else:  # second half of S2
        pd_all = 1
        pd = 1
        for i in NC2:
            all_PrDs.append(pd_all)
            all_occ.append(i)
            if i >= p.PDsizeThL:
                thresh_PrDs.append(pd)
                if i > p.PDsizeThH:
                    thresh_occ.append(p.PDsizeThH)
                else:
                    thresh_occ.append(i)
                pd += 1
            pd_all += 1

    # read points from tesselated sphere:
    PrD_map1_eul = pandas.read_csv(p.ref_ang_file1, delimiter='\t')

    all_phi = PrD_map1_eul['phi']
    all_theta = PrD_map1_eul['theta']

    # ad hoc ratios to make sure S2 volume doesn't freeze-out due to too many particles to plot:
    ratio = float(sum(all_occ)) / 5000
    S2_density_all = [5, 10, 25, 50, 100, 250, 500, 1000, 10000, 100000]
    S2_density_all = list(filter(lambda a: a < int(sum(all_occ)), S2_density_all))
    S2_density_all = list(filter(lambda a: a > int(ratio), S2_density_all))

    return [int(el) for el in S2_density_all]
