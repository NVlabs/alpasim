# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pickle
from multiprocessing import Manager, Pool
from typing import Callable, Iterable, List, Optional

from tqdm import tqdm


def parallel_apply(
    element_fn: Callable,
    element_list: Iterable,
    num_workers: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> List:
    return list(parallel_iapply(element_fn, element_list, num_workers, desc, disable))


def parallel_iapply(
    element_fn: Callable,
    element_list: Iterable,
    num_workers: int,
    desc: Optional[str] = None,
    disable: bool = False,
) -> Iterable:
    with Pool(processes=num_workers) as pool:
        for fn_output in tqdm(
            pool.imap(element_fn, element_list),
            desc=desc,
            total=len(element_list),
            disable=disable,
        ):
            yield fn_output


def pickle_objects(objs: List) -> List[bytes]:
    pickled_objs: List[bytes] = list()
    for obj in objs:
        pickled_objs.append(pickle.dumps(obj))

    return pickled_objs


class AsyncExecutor:
    def __init__(
        self, num_workers: int, total_jobs: int, desc: str, position: int, disable: bool
    ):
        self.pool = Pool(processes=num_workers)
        self.manager = Manager()
        self.results_queue = self.manager.Queue()

        self.pbar = tqdm(
            desc=desc, position=position, total=total_jobs, disable=disable
        )

    def error(self, err):
        raise err

    def prompt(self, results: List[bytes]):
        self.pbar.update(len(results))
        for result in results:
            self.results_queue.put(result)

    def schedule(self, function, args):
        return self.pool.apply_async(
            function, args, callback=self.prompt, error_callback=self.error
        )

    def wait(self):
        self.pool.close()
        self.pool.join()
        self.pbar.close()
