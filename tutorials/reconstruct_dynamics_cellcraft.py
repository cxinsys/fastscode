import sys
import os
import os.path as osp

import numpy as np

import fastscode as fs

if __name__ == "__main__":
    droot = osp.abspath(sys.argv[1])
    dpath_init = osp.join(droot, sys.argv[2])
    dpath_rm = osp.join(droot, sys.argv[3])

    fpath_out = osp.join(droot, sys.argv[8])

    backend = sys.argv[4]
    num_device = sys.argv[5]
    length = sys.argv[6]

    batch_size = sys.argv[7]

    init = np.loadtxt(dpath_init, delimiter="\t", usecols=[1],)
    result_matrix = np.loadtxt(dpath_rm, delimiter="\t")

    worker = fs.Simulator(init=init,
                          result_matrix=result_matrix,
                          num_times=length,
                          fpath_out=fpath_out)

    result = worker.run(backend=backend,
                        device_ids=num_device,
                        batch_size=batch_size)