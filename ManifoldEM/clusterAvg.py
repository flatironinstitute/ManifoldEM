import numpy as np

from ManifoldEM import myio, p
'''
Copyright (c) Columbia University Evan Seitz 2019
'''


def op(clust, PrD):
    dist_file = '{}prD_{}'.format(p.dist_file, PrD)
    data = myio.fin1(dist_file)

    imgAll = data['imgAll']
    imgSize = np.shape(imgAll)[0]
    boxSize = np.shape(imgAll)[1]

    imgAvg = np.zeros(shape=(boxSize, boxSize), dtype=float)

    for i in clust:
        imgAvg += imgAll[i]

    return imgAvg