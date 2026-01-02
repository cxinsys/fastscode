import os
import os.path as osp
import time
import math
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue
import queue

import numpy as np

from mate.array import get_array_module
from mate.utils import get_device_list

from fastscode.utils import compute_chunk_size, check_gpu_computability, save_results
from fastscode.worker import WorkerProcess


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
                 dtype=np.float32,
                 use_binary=False
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
        self.use_binary = use_binary

        print("[Num. genes: {}, Num. cells: {}]".format(self.num_tf, self.num_cell))

    @property
    def am(self):
        return self._am

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size_b=1,
            batch_size=None,
            chunk_size=None,
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
        new_b = np.random.uniform(low=self.min_b, high=self.max_b, size=(batch_size_b, self.num_z)).astype(
            np.float32)  # (B, p)
        old_b = np.zeros(new_b.shape[-1], dtype=new_b.dtype)  # (p)

        if not batch_size:
            batch_size = len(self.exp_data)

        gpu_batch = np.ceil(len(self.exp_data) / (len(device_ids) * procs_per_device)).astype(np.int32)

        if not chunk_size:
            chunk_size = compute_chunk_size(batch=batch_size,
                                                    C=len(self.pseudotime),
                                                    sb=batch_size_b,
                                                    D=new_b.shape[-1],
                                                    dtype=self.dtype)

        multiprocessing.set_start_method('spawn', force=True)

        task_queues = []
        result_queue = Queue()

        worker_data = []
        for j, start in enumerate(range(0, len(self.exp_data), gpu_batch)):
            end = start + gpu_batch
            worker_data.append({
                'backend': backend + ":" + str(device_ids[j % len(device_ids)]),
                'exp_data': self.exp_data[start:end, :],
                'pseudotime': self.pseudotime,
                'batch_size': batch_size,
                'dtype': self.dtype
            })

        workers = []
        num_workers = len(worker_data) * procs_per_device

        for i in range(num_workers):
            task_queue = Queue()
            task_queues.append(task_queue)

            worker_idx = i % len(worker_data)
            worker = WorkerProcess(
                worker_id=i,
                backend=worker_data[worker_idx]['backend'],
                exp_data=worker_data[worker_idx]['exp_data'],
                pseudotime=worker_data[worker_idx]['pseudotime'],
                batch_size=worker_data[worker_idx]['batch_size'],
                chunk_size=chunk_size,
                dtype=worker_data[worker_idx]['dtype'],
                task_queue=task_queue,
                result_queue=result_queue,
                sb=batch_size_b,
                D=new_b.shape[-1]
            )

            workers.append(worker)
            worker.start()

        print("[DEVICE: {}, Num. GPUS: {}, Process per device: {}, Sampling Batch: {}, Batch Size: {}, Chunk Size: {}]"
              .format(backend, len(device_ids), procs_per_device, batch_size_b, batch_size, chunk_size))

        try:
            list_W = []
            pbar = tqdm(range(1, self.max_iter + 1))
            for i in pbar:
                pbar.set_description("[ITER] {}/{}, [Num. Sampling] {}".format(i, self.max_iter, i * batch_size_b))
                target = np.random.randint(0, self.num_z, size=batch_size_b)
                new_b[np.arange(len(new_b)), target] = np.random.uniform(low=self.min_b, high=self.max_b,
                                                                         size=batch_size_b)

                if i == self.max_iter:
                    new_b = old_b.copy()
                    new_b = new_b.reshape(1, -1)

                # Send tasks to workers
                for j, task_queue in enumerate(task_queues):
                    task_queue.put((i, new_b))

                # Collect results
                tmp_rss = np.zeros(len(new_b))
                collected_results = 0

                while collected_results < num_workers:
                    try:
                        worker_id, result = result_queue.get(timeout=30)
                        if isinstance(result, str) and result.startswith("ERROR"):
                            print(f"Worker {worker_id} error: {result}")
                            continue

                        part_rss, W_batch = result
                        tmp_rss += part_rss

                        if i == self.max_iter:
                            list_W.extend(W_batch)

                        collected_results += 1

                    except queue.Empty:
                        print(f"Timeout waiting for results at iteration {i}")
                        break

                if i == self.max_iter and list_W:
                    W = np.concatenate(list_W, axis=1)[0]  # (tf, p)

                local_min = np.min(tmp_rss)
                inds_min = np.argmin(tmp_rss)
                if local_min < RSS:
                    RSS = local_min
                    old_b = new_b[inds_min].copy()  # (p)
                else:
                    new_b = np.tile(old_b.copy(), len(new_b)).reshape(len(new_b), -1)  # (b, p)

        finally:
            # Stop all workers
            for task_queue in task_queues:
                task_queue.put("STOP")

            # Wait for workers to finish
            for worker in workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()

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
            save_results(droot=self.droot, rss=RSS, A=A,
                         node_name=self.node_name, use_binary=self.use_binary)

        return RSS, A