import os
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
    parser.add_argument('--batch_size_b', type=int, dest='sb', required=False, default=100)
    parser.add_argument('--batch_size', type=int, dest='batch_size', required=False, default=100)
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

    exp_data = np.loadtxt(dpath_exp_data, delimiter=",", dtype=str)  # cell x gene
    node_name = exp_data[0, 1:]
    exp_data = exp_data[1:, 1:].astype(np.float64).T  # gene x cell

    # # min-max norm
    # exp_data = (exp_data - np.min(exp_data, axis=1)[:, None]) / \
    #            (np.max(exp_data, axis=1) - np.min(exp_data, axis=1))[:, None]
    pseudotime = np.loadtxt(dpath_trj_data, delimiter="\t")
    branch = np.loadtxt(dpath_branch_data, delimiter="\t")

    pseudotime = pseudotime[branch == 1]
    exp_data = exp_data[:, branch == 1]

    repeats = np.arange(args.repeat, dtype=np.int32)

    scores = np.zeros((len(exp_data), len(exp_data)), dtype=np.float64)
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
                              dtype=np.float32)

        rss, score_matrix = worker.run(backend=backend,
                                  device_ids=num_devices,
                                  batch_size_b=sb,
                                  batch_size=batch_size)

        scores += score_matrix

    mean_A = scores / args.repeat
    tmp_rm = np.concatenate([node_name[:, None], mean_A.astype(str)], axis=1)
    extended_nn = np.concatenate((['Score'], node_name))
    tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])
    np.save(os.path.join(droot, f"score_sb-{sb}_batch-{batch_size}_z-{num_z}_iter-{max_iter}_repeat-{args.repeat}.npy"), tmp_rm)

    execution_time = time.time() - s_time

    with open(fout_stats, 'a') as f:
        f.write("%s,%s,%s,%d,%d,%d,%d,%d,%d,%f\n" % (platform.node().upper(), dataset, backend,
                                                  num_devices, sb, batch_size,
                                                  num_z, max_iter, args.repeat, execution_time))


