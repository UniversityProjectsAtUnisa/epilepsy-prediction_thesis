import torch
from multiprocessing import Pool


class ConditionalParallelTrainer:
    def __init__(self, train_function, parallel_training: bool = False, n_workers: int = 3):
        self.parallel_training = parallel_training
        self.n_workers = n_workers
        self.train_function = train_function

    def __call__(self, *args, **kw):
        if self.parallel_training:
            torch.multiprocessing.set_start_method('spawn')
            with Pool(self.n_workers) as p:
                p.starmap(self.train_function, *args, **kw)
        else:
            self.train_function(*args, **kw)
