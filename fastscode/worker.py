import os
from multiprocessing import Process
import queue

import numpy as np

from mate.array import get_array_module


class WorkerProcess(Process):
    def __init__(self, worker_id, backend, exp_data, pseudotime, batch_size, dtype, task_queue, result_queue):
        super().__init__()
        self.worker_id = worker_id
        self.backend = backend
        self.exp_data = exp_data
        self.pseudotime = pseudotime
        self.batch_size = batch_size
        self.dtype = dtype
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.am = None

    def setup_backend(self):
        if self.backend.startswith('tf') or self.backend.startswith('tensorflow'):
            device_id = self.backend.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        self.am = get_array_module(self.backend)

        # Convert data to appropriate array format
        self.X = self.am.array(self.exp_data, dtype=self.dtype)
        self.pseudotime_array = self.am.array(self.pseudotime, dtype=self.dtype)

    def estimate_W_worker(self, new_b):
        new_b = self.am.array(new_b, dtype=self.dtype)  # (sb, p)

        noise = self.am.random_uniform(
            low=-0.001, high=0.001,
            size=(len(new_b), new_b.shape[-1], len(self.pseudotime))
        )  # (sb, p, c)

        Z = self.am.exp(self.am.dot(new_b[..., None], self.pseudotime_array[None, :])) + self.am.astype(noise,
                                                                                                        self.dtype)  # (sb, p, c)
        ZZt = self.am.matmul(Z, self.am.transpose(Z, axes=(0, 2, 1)))  # (sb, p, p)

        partsum_rss = np.zeros(len(new_b))
        list_W = []

        for i, start in enumerate(range(0, len(self.X), self.batch_size)):
            end = start + self.batch_size

            batch_X = self.X[start:end]
            ZX = self.am.matmul(Z, self.am.transpose(batch_X, axes=(1, 0)))  # (sb, p, g)

            try:
                W = self.am.linalg_solve(ZZt, ZX)  # (sb, p, g)
            except:
                W = self.am.matmul(self.am.pinv(ZZt), ZX)  # (sb, p, g)

            W = self.am.transpose(W, axes=(0, 2, 1))  # (sb, g, p)
            WZ = self.am.matmul(W, Z)  # (sb, g, c)
            diffs = (batch_X - WZ) ** 2
            tmp_rss = self.am.sum(diffs, axis=(1, 2))  # (sb)

            partsum_rss += self.am.asnumpy(tmp_rss)  # (sb)
            list_W.append(self.am.asnumpy(W))

        return partsum_rss, list_W

    def run(self):
        self.setup_backend()

        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task == "STOP":
                    break

                iteration, new_b = task
                result = self.estimate_W_worker(new_b)
                self.result_queue.put((self.worker_id, result))

            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put((self.worker_id, f"ERROR: {e}"))
