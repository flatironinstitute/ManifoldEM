import numpy as np

from ManifoldEM.trajectory import concatenate_bin


def _write_bin(out_dir, taubins):
    concatenate_bin(0, nlsa_movies=[np.ones((5, 4, 4), dtype=np.float16)], taubins_by_prd=[[taubins]],
                    phis_by_prd=[[np.zeros(len(taubins))]], thetas_by_prd=[[np.zeros(len(taubins))]],
                    psis_by_prd=[[np.zeros(len(taubins))]], states_per_coord=50, con_orders=[1],
                    pixel_size=1.0, out_dir=str(out_dir), traj_name="1")


def test_a_state_without_particles_writes_no_files(tmp_path):
    _write_bin(tmp_path, np.array([1, 3]))
    # main's names, with traj_name
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EulerAngles_1_1_of_50.star", "imgsRELION_1_1_of_50.mrcs"]

    # a rerun in which no particle falls in the state removes the earlier files instead of failing
    _write_bin(tmp_path, np.array([], dtype=int))
    assert list(tmp_path.iterdir()) == []
