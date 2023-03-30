'''
Copyright (c) Columbia University Evan Seitz 2019    
Copyright (c) Columbia University Hstau Liao 2019    
'''

import os
import pickle

from ManifoldEM import myio, p


def op(do):  # p.tess_file is known
    if do == 1:  #via 'set_params.op(1)': to retrieve data stored in parameters file
        p.load(f'params_{p.proj_name}.toml')
    elif do == 0:  #via 'set_params.op(0)': to update values within parameters file
        p.save(f'params_{p.proj_name}.toml')
    elif do == -1:  #via 'set_params.op(-1)': to print all values in parameters file
        print(p.todict())
