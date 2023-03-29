import time

from ManifoldEM import p
'''
Copyright (c) UWM, Ali Dashti 2016 (original matlab version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
Copyright (c) Columbia University Evan Seitz 2019 (python version)

'''


def op(IMG1, prD, psinum, fps):
    import os
    import imageio
    import numpy as np

    dim = int(np.floor(np.sqrt(max(IMG1.shape))))  # window size
    nframes = IMG1.shape[1]
    images = -IMG1
    gif_path = os.path.join(p.out_dir, "topos", f"PrD_{prD + 1}", f'psi_{psinum + 1}.gif')
    frame_dt = 1.0/fps
    with imageio.get_writer(gif_path, mode='I', duration=frame_dt) as writer:
        for i in range(nframes):
            img = images[:, i].reshape(dim, dim)
            frame = np.round(255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)

            frame_path = p.out_dir + '/topos/PrD_{}/psi_{}/frame{:02d}.png'.format(prD + 1, psinum + 1, i)
            imageio.imwrite(frame_path, frame)
            writer.append_data(frame)
