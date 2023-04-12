import pickle
from ManifoldEM.util import debug_print
'''
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''


def fin1(filename):
    with open(filename, 'rb') as f:
        try:
            data = pickle.load(f)
            f.close()
            return data
        except Exception as e:
            debug_print(str(e))
            return None


def fout1(filename, **kwargs):
    with open(filename, 'wb') as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
