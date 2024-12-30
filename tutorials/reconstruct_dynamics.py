import os
import os.path as osp
import argparse

import numpy as np

import fastscode as fs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter parser')
    parser.add_argument('--droot', type=str, dest='droot', required=False, default=osp.abspath('./'))
    parser.add_argument('--fp_init', type=str, dest='fp_init', required=True)
    parser.add_argument('--fp_rm', type=str, dest='fp_rm', required=True)
    parser.add_argument('--backend', type=str, dest='backend', required=False, default='cpu')
    parser.add_argument('--num_device', type=int, dest='num_device', required=False, default=1)
    parser.add_argument('--length', type=int, dest='length', required=False, default=100)
    parser.add_argument('--batch_size', type=int, dest='batch_size', required=False, default=100)
    parser.add_argument('--fpath_out', type=str, dest='fpath_out', required=False)

    args = parser.parse_args()

    droot = osp.abspath(args.droot)
    dpath_init = osp.join(droot, args.fp_init)
    dpath_rm = osp.join(droot, args.fp_rm)

    fpath_out = osp.join(droot, args.fpath_out)

    backend = args.backend
    num_device = args.num_device
    length = args.length

    batch_size =args.batch_size

    # init = np.loadtxt(dpath_init, delimiter="\t", usecols=[1],)
    result_matrix = np.loadtxt(dpath_rm, delimiter="\t")
    init = np.random.rand(len(result_matrix))

    worker = fs.Simulator(init=init,
                          result_matrix=result_matrix,
                          num_times=length,
                          fpath_out=fpath_out)

    result = worker.run(backend=backend,
                        device_ids=num_device,
                        batch_size=batch_size)
