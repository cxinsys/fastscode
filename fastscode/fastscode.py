import os
import os.path as osp
import time
import math
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np

from mate.array import get_array_module
from mate.utils import get_device_list, istarmap

from fastscode.utils import calculate_batchsize, check_gpu_computability, save_results

class FastSCODE(object):
    def __init__(self,
                 dpath_exp_data=None,
                 dpath_trj_data=None,
                 droot=None,
                 exp_data=None,
                 node_name=None,
                 pseudotime=None,
                 num_tf=None,
                 num_cell=None,
                 num_z=4,
                 max_iter=100,
                 max_b=2.0,
                 min_b=-10.0,
                 dtype=np.float32
                 ):

        self.exp_data = None
        self.node_name = None
        self.pseudotime = None

        if exp_data is not None:
            self.exp_data = exp_data  # (gene, cell)

            if pseudotime is None:
                raise ValueError("pseudotime data should be defined if using expression data directly")

            self.pseudotime = pseudotime  # (cell)
            self.node_name = node_name
        else:
            if not dpath_exp_data or not dpath_exp_data:
                raise ValueError("One of the following variable is not defined correctly: "
                                 "dpath_exp_data, dpath_trj_data")

            exp_data = np.loadtxt(dpath_exp_data, delimiter=",", dtype=str)
            self.node_name = exp_data[0, 1:]
            self.exp_data = exp_data[1:, 1:].astype(dtype).T
            self.pseudotime = np.loadtxt(dpath_trj_data, delimiter="\t")
            self.pseudotime = self.pseudotime[:, 1]

        self.num_tf = len(self.exp_data)
        self.num_cell = len(self.pseudotime)
        self.num_z = num_z
        self.max_iter = max_iter
        self.max_b = max_b
        self.min_b = min_b
        self.dtype = dtype

        if num_tf is not None:
            self.num_tf = num_tf
        if num_cell is not None:
            self.num_cell = num_cell

        self.exp_data = self.exp_data[:self.num_tf, :self.num_cell].astype(dtype)
        self.pseudotime = self.pseudotime[:self.num_cell].astype(dtype)
        self.pseudotime = self.pseudotime / np.max(self.pseudotime)

        self.droot = droot

        print("[Num. genes: {}, Num. cells: {}]".format(self.num_tf, self.num_cell))

    @property
    def am(self):
        return self._am

    def estimateW(self,
                  backend='cpu',
                  exp_data=None,
                  pseudotime=None,
                  new_b=None,
                  batch_size=None,
                  id=0,
                  dtype=np.float64):

        # if backend.startswith('tf') or backend.startswith('tensorflow'):
        #     import tensorflow as tf
        #     device_id = backend.split(":")[-1]
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        #
        #     gpus = tf.config.experimental.list_physical_devices('GPU')
        #     if gpus:
        #         try:
        #             for gpu in gpus:
        #                 tf.config.experimental.set_memory_growth(gpu, True)
        #         except RuntimeError as e:
        #             print(f"TensorFlow GPU 설정 오류: {e}")
        # elif backend.startswith('jax'):
        #     device_id = backend.split(":")[-1]
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        #     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 사전 할당 비활성화
        #
        #     import jax
        #     jax.config.update('jax_platform_name', 'gpu')

        am = get_array_module(backend)

        if backend.startswith('tf') or backend.startswith('tensorflow'):
            device_id = backend.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        X = am.array(exp_data, dtype=dtype)  # (outer_batch, cell)
        pseudotime = am.array(pseudotime, dtype=dtype)  # (c)
        new_b = am.array(new_b, dtype=dtype)  # (sb, p)

        noise = am.random_uniform(low=-0.001, high=0.001, size=(len(new_b), new_b.shape[-1], len(pseudotime)))  # (sb, p, c)
        Z = am.exp(am.dot(new_b[..., None], pseudotime[None, :])) + am.astype(noise, dtype=dtype)  # (sb, p, c)

        ZZt = am.matmul(Z, am.transpose(Z, axes=(0, 2, 1)))  # (sb, p, p)

        partsum_rss = np.zeros(len(new_b))
        list_W = []

        for i, start in enumerate(range(0, len(X), batch_size)):
            end = start + batch_size

            batch_X = X[start:end]
            ZX = am.matmul(Z, am.transpose(batch_X, axes=(1, 0)))  # (sb, p, g)

            try:
                W = am.linalg_solve(ZZt, ZX)  # (sb, p, g)
            except:
                W = am.matmul(am.pinv(ZZt), ZX)  # (sb, p, g)

            W = am.transpose(W, axes=(0, 2, 1))  # (sb, g, p)
            WZ = am.matmul(W, Z)  # (sb, g, c)
            diffs = (batch_X - WZ) ** 2
            tmp_rss = am.sum(diffs, axis=(1, 2))  # (sb)

            partsum_rss += am.asnumpy(tmp_rss)  # (sb)
            list_W.append(am.asnumpy(W))

        return partsum_rss, list_W

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size_b=1,
            batch_size=None,
            seed=None
            ):

        if not backend:
            backend = 'cpu'

        if not device_ids:
            if backend == 'cpu':
                device_ids = [0]
            else:
                device_ids = get_device_list()

        if not procs_per_device:
            procs_per_device = 1

        if type(device_ids) is int:
            list_device_ids = [x for x in range(device_ids)]
            device_ids = list_device_ids

        self._am = get_array_module(backend + ":" + str(device_ids[0]))

        if not seed:
            np.random.seed(int(datetime.now().timestamp()))
            self.am.seed(int(datetime.now().timestamp()))
        else:
            np.random.seed(seed)
            self.am.seed(seed)

        RSS = np.inf

        W = None
        new_b = np.random.uniform(low=self.min_b, high=self.max_b, size=(batch_size_b, self.num_z)).astype(np.float32) # (B, p)
        old_b = np.zeros(new_b.shape[-1], dtype=new_b.dtype)  # (p)

        if not batch_size:
            batch_size = len(self.exp_data)

        # print("[Batch Size auto calculated]")
        # outer_batch = calculate_batchsize(batch=batch_size,
        #                                 exp_data_shape=self.exp_data.shape,
        #                                 new_b_shape=new_b.shape,
        #                                 dtype=self.dtype,
        #                                 num_gpus=len(device_ids),
        #                                 num_ppd=procs_per_device)
        outer_batch = np.ceil(len(self.exp_data) / (len(device_ids) * procs_per_device)).astype(np.int32)


        multiprocessing.set_start_method('spawn', force=True)

        list_W = []
        list_backend = []
        list_data = []
        list_time = []
        list_batch = []
        list_dtype = []
        list_id = []

        for j, start in enumerate(range(0, len(self.exp_data), outer_batch)):
            end = start + outer_batch

            list_backend.append(backend + ":" + str(device_ids[j % len(device_ids)]))
            list_data.append(self.exp_data[start:end, :])
            list_time.append(self.pseudotime)
            list_batch.append(batch_size)
            list_id.append(j)
            list_dtype.append(self.dtype)

        print("[DEVICE: {}, Num. GPUS: {}, Process per device: {}, Sampling Batch: {}, Batch Size: {}]"
              .format(backend, len(device_ids), procs_per_device, batch_size_b, batch_size))

        with multiprocessing.Pool(processes=len(list_backend) * procs_per_device) as pool:
            pbar = tqdm(range(1, self.max_iter + 1))
            for i in pbar:
                pbar.set_description("[ITER] {}/{}, [Num. Sampling] {}".format(i, self.max_iter, i*batch_size_b))
                target = np.random.randint(0, self.num_z, size=batch_size_b)
                new_b[np.arange(len(new_b)), target] = np.random.uniform(low=self.min_b, high=self.max_b, size=batch_size_b)

                if i == self.max_iter:
                    new_b = old_b.copy()
                    new_b = new_b.reshape(1, -1)

                tmp_rss = np.zeros(len(new_b))

                list_newb = []
                for k in enumerate(range(0, len(list_backend))):
                    list_newb.append(new_b)

                inputs = zip(list_backend, list_data, list_time, list_newb, list_batch, list_id, list_dtype)

                for batch_result in pool.istarmap(self.estimateW, inputs):
                    part_rss, W = batch_result
                    tmp_rss += part_rss

                    if i == self.max_iter:
                        list_W.extend(W)
                        W = np.concatenate(list_W, axis=1)[0]  # (tf, p)

                local_min = np.min(tmp_rss)
                inds_min = np.argmin(tmp_rss)
                if local_min < RSS:
                    RSS = local_min
                    old_b = new_b[inds_min].copy()  # (p)
                else:
                    new_b = np.tile(old_b.copy(), len(new_b)).reshape(len(new_b), -1)  # (b, p)

        # after iterating
        if check_gpu_computability(w_shape=W.shape, new_b_shape=new_b[0].shape, dtype=self.dtype):
            new_b = self.am.array(new_b[0], dtype=self.dtype)  # (p)
            W = self.am.array(W, dtype=self.dtype)  # (tf, p)

            b_matrix = self.am.diag(new_b)  # (p, p)
            invW = self.am.pinv(W)  # (p, tf)
            A = self.am.dot(self.am.dot(W, b_matrix), invW)  # (tf, p) (p) (p, tf)

            W = self.am.asnumpy(W)
            A = self.am.asnumpy(A)
            b_matrix = self.am.asnumpy(b_matrix)
        else:
            b_matrix = np.diag(new_b[0])
            invW = np.linalg.pinv(W)
            A = W @ b_matrix @ invW

        if self.droot is not None:
            save_results(droot=self.droot, rss=RSS, W=W, A=A, B=b_matrix, node_name=self.node_name)

        return RSS, A