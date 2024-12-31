import os
import sys
import os.path as osp
import argparse

import numpy as np

import fastscode as fs

if __name__ == "__main__":
    droot = osp.abspath(sys.argv[1])  # droot
    dpath_exp_data = osp.join(droot, sys.argv[2])  # expression data
    dpath_trj_data = osp.join(droot, sys.argv[3])  # pseudotime data

    num_z = int(sys.argv[4])  # number of z
    max_iter = int(sys.argv[5])  # number of iteration

    backend = sys.argv[6]  # backend, 'gpu', 'torch', etc..
    num_devices = int(sys.argv[7])  # number of devices
    sb = int(sys.argv[8])  # sampling batch size
    chunk_size = int(sys.argv[9])  # chunk size

    spath_droot = osp.join(droot, sys.argv[10])  # save path

    exp_data = np.loadtxt(dpath_exp_data, delimiter=",", dtype=str)
    node_name = exp_data[0, 1:]
    exp_data = exp_data[1:, 1:].astype(np.float64).T
    pseudotime = np.loadtxt(dpath_trj_data, delimiter="\t")

    repeats = np.arange(int(sys.argv[11]), dtype=np.int32)  # num repeat

    As = []
    for r in repeats:
        spath_droot_r = osp.join(spath_droot, str(r))

        worker = fs.FastSCODE(exp_data=exp_data,
                              pseudotime=pseudotime,
                              node_name=node_name,
                              droot=spath_droot_r,
                              num_tf=None,
                              num_cell=None,
                              num_z=num_z,
                              max_iter=max_iter,
                              dtype=np.float64)

        rss, W, A, B = worker.run(backend=backend,
                                  device_ids=num_devices,
                                  sampling_batch=sb,
                                  chunk_size=chunk_size)

        As.append(A)

    mean_A = np.mean(As, axis=0)
    tmp_rm = np.concatenate([node_name[:, None], mean_A.astype(str)], axis=1)
    extended_nn = np.concatenate((['TE'], node_name))
    tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])
    np.savetxt(os.path.join(droot, "meanA.txt"), tmp_rm, delimiter="\t", fmt="%s")

    # RSS, W, A and B will save under the droot directory
