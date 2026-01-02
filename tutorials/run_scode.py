import os
import gc
import os.path as osp
import argparse
import time
import platform

import numpy as np

import fastscode as fs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameter parser')
    parser.add_argument('--droot', type=str, dest='droot', required=False, default=osp.abspath('./'))
    parser.add_argument('--fp_exp', type=str, dest='fp_exp', required=True)
    parser.add_argument('--fp_trj', type=str, dest='fp_trj', required=True)
    parser.add_argument('--fp_branch', type=str, dest='fp_branch', required=True)
    parser.add_argument('--num_z', type=int, dest='num_z', required=False, default=10)
    parser.add_argument('--max_iter', type=int, dest='max_iter', required=False, default=100)
    parser.add_argument('--backend', type=str, dest='backend', required=False, default='gpu')
    parser.add_argument('--num_devices', type=int, dest='num_devices', required=False, default=1)
    parser.add_argument('--ppd', type=int, required=False, default=1)
    parser.add_argument('--batch_size_b', type=int, dest='sb', required=False, default=100)
    parser.add_argument('--batch_size', type=int, dest='batch_size', required=False, default=None)
    parser.add_argument('--chunk_size', type=int, dest='chunk_size', required=False, default=None)
    parser.add_argument('--sp_droot', type=str, dest='sp_droot', required=False, default='None')
    parser.add_argument('--num_repeat', type=int, dest='repeat', required=False, default=1)

    args = parser.parse_args()

    s_time = time.time()

    fout_stats = 'execution_time_fastscode.csv'
    if not osp.exists(fout_stats):
        with open(fout_stats, 'w') as f:
            f.write("Hostname,Dataset,Backend,Num. devices," \
                    "Sampling batch,Batch size," \
                    "Num. Z,Num. iter," \
                    "Num. repeat,Execution time\n")

    droot = osp.abspath(args.droot)
    dpath_exp_data = osp.join(droot, args.fp_exp)
    dpath_trj_data = osp.join(droot, args.fp_trj)
    dpath_branch_data = osp.join(droot, args.fp_branch)

    num_z = args.num_z
    max_iter = args.max_iter

    dataset = args.fp_exp.split('.')[0]

    if args.sp_droot != 'None':
        spath_droot = osp.join(droot, args.sp_droot)
    else:
        spath_droot = None

    backend = args.backend
    num_devices = args.num_devices
    sb = args.sb
    batch_size = args.batch_size
    chunk_size = args.chunk_size

    print("Loading..")
    with open(dpath_exp_data, 'r') as f:
        node_name = np.array(f.readline().strip().split(',')[1:]).astype(str)

    exp_data = np.loadtxt(dpath_exp_data, delimiter=',',
                          skiprows=1, usecols=range(1, len(node_name) + 1), dtype=np.float32).T

    # # min-max norm
    # exp_data = (exp_data - np.min(exp_data, axis=1)[:, None]) / \
    #            (np.max(exp_data, axis=1) - np.min(exp_data, axis=1))[:, None]
    pseudotime = np.loadtxt(dpath_trj_data, delimiter="\t")
    branch = np.loadtxt(dpath_branch_data, delimiter="\t")

    pseudotime = pseudotime[branch == 1]
    exp_data = exp_data[:, branch == 1]

    repeats = np.arange(args.repeat, dtype=np.int32)

    scores = np.zeros((len(exp_data), len(exp_data)), dtype=np.float32)
    gc.collect()

    for r in repeats:
        if spath_droot is not None:
            spath_droot_r = osp.join(spath_droot, str(r))
        else:
            spath_droot_r = None

        worker = fs.FastSCODE(exp_data=exp_data,
                              pseudotime=pseudotime,
                              node_name=node_name,
                              droot=spath_droot_r,
                              num_tf=None,
                              num_cell=None,
                              num_z=num_z,
                              max_iter=max_iter,
                              dtype=np.float32,
                              use_binary=True)

        rss, score_matrix = worker.run(backend=backend,
                                       device_ids=num_devices,
                                       procs_per_device=args.ppd,
                                       batch_size_b=sb,
                                       batch_size=batch_size,
                                       chunk_size=chunk_size)

        scores += score_matrix

    s1time = time.time()
    mean_A = scores / args.repeat
    # tmp_rm = np.concatenate([node_name[:, None], mean_A.astype(str)], axis=1)
    # extended_nn = np.concatenate((['Score'], node_name))
    # tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])

    np.savetxt(os.path.join(spath_droot, "node_name.txt"), node_name, delimiter="\t", fmt="%s")

    np.save(os.path.join(spath_droot, "avg_score_matrix.npy"), mean_A)

    print("Elapsed time for saving matrix: ", time.time() - s1time)

    execution_time = time.time() - s_time

    if chunk_size is None:
        chunk_size = len(exp_data)
    with open(fout_stats, 'a') as f:
        f.write("%s,%s,%s,%d,%d,%d,%d,%d,%d,%f\n" % (platform.node().upper(), dataset, backend,
                                                     num_devices, sb, chunk_size,
                                                     num_z, max_iter, args.repeat, execution_time))