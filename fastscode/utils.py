import os
import multiprocessing.pool as mpp

try:
    import torch
except (ModuleNotFoundError, ImportError) as err:
    pass

import numpy as np


def save_results(droot, rss, A, node_name, use_binary=True):
    os.makedirs(droot, exist_ok=True)
    np.savetxt(os.path.join(droot, "RSS.txt"), [rss], delimiter="\t", fmt="%.14f")

    if use_binary:
        np.savetxt(os.path.join(droot, "node_name.txt"), node_name, delimiter="\t", fmt="%s")
        np.save(os.path.join(droot, "score_matrix.npy"), A)
    else:
        tmp_rm = np.concatenate([node_name[:, None], A.astype(str)], axis=1)
        extended_nn = np.concatenate((['Score'], node_name))
        tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])

        np.savetxt(os.path.join(droot, "score_matrix.txt"), tmp_rm, delimiter="\t", fmt="%s")


def calculate_lm_memory_usage(batch, exp_data_shape, new_b_shape, dtype=np.float32):
    dtype_size = np.dtype(dtype).itemsize

    g, c = exp_data_shape # (g: gene, c: cell)
    s, z = new_b_shape  # (s: sampling batch, z: latent factor)

    exp_data_memory = batch * c * dtype_size  # exp_data
    pseudotime_memory = c * dtype_size  # pseudotime
    new_b_memory = s * z * dtype_size  # new_b

    Z_memory = s * z * c * dtype_size  # Z
    XtX_memory = s * z * z * dtype_size  # XtX
    Xty_memory = s * z * batch * dtype_size  # Xty
    W_memory = s * batch * z * dtype_size  # W
    WZ_memory = s * batch * c * dtype_size  # WZ
    diffs_memory = s * batch * c * dtype_size  # diffs

    fix_memory = (pseudotime_memory + new_b_memory + Z_memory + XtX_memory) / (1024 ** 2)
    other = (exp_data_memory + Xty_memory + W_memory + WZ_memory + diffs_memory + diffs_memory) / (1024 ** 2)

    return fix_memory, other


def compute_chunk_size(batch, C, sb, D, dtype=np.float32):
    size = np.dtype(dtype).itemsize

    # X: batch x C, pseudotime: C, new_b: sb x D, noise: sb x D x C, Z: sb x D x C, ZZt: sb x D x D
    init_mem = ((batch + 1) * C + (1 + 2 * C + D) * D * sb) * size

    gpu_mem = get_gpu_memory(0)

    remain_mem = gpu_mem - init_mem

    # batch_X: chunk x C, ZX: sb x D x chunk, W: sb x D x chunk, Wt: sb x chunk x D, WZ: sb x chunk x C, diffs: sb x chunk x C, tmp_rss: sb
    chunk = int(remain_mem // ((3 * sb * D + (1 + 2 * sb) * C + sb) * size))

    if chunk < 1:
        raise ValueError("The batch size is too large")

    if chunk > batch:
        chunk = batch

    return chunk


def check_gpu_computability(w_shape, new_b_shape, dtype=np.float32):
    dtype_size = np.dtype(dtype).itemsize

    g, z = w_shape  # (g: gene, c: cell)
    z = new_b_shape[-1]  # (z: latent factor)

    b_matrix_memory = z * z * dtype_size  # exp_data
    w_memory = g * z * dtype_size  # pseudotime

    inv_w_memory = z * g * dtype_size  # Z

    total_memory = (b_matrix_memory + w_memory + inv_w_memory) / (1024 ** 2)

    gpu_mem = get_gpu_memory(gpu_index=0)

    return total_memory < gpu_mem


def get_gpu_memory(gpu_index=0):
    torch.cuda.set_device(gpu_index)
    total_memory = torch.cuda.get_device_properties(gpu_index).total_memory

    return 0.7 * total_memory


def calculate_batchsize(batch,
                        exp_data_shape,
                        new_b_shape,
                        dtype=np.float64,
                        num_gpus=1,
                        num_ppd=1):
    fix, other = calculate_lm_memory_usage(batch=batch,
                                           exp_data_shape=exp_data_shape,
                                           new_b_shape=new_b_shape,
                                           dtype=dtype)
    gpu_mem = get_gpu_memory(gpu_index=0)

    free_mem = gpu_mem - (fix * num_ppd)

    if free_mem < 0:
        raise ValueError("The batch size or procs_per_device value is too large.")

    remain_mem = free_mem - other

    dtype_size = np.dtype(dtype).itemsize

    single_node_mem = exp_data_shape[1] * dtype_size / (1024 ** 2)
    outer_batch = np.ceil(remain_mem / (single_node_mem * num_gpus)).astype(np.int32)

    if outer_batch < 0:
        raise ValueError("The number of processors you want to use is too many for the batch size. ")

    return outer_batch


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap