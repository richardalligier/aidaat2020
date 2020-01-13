import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler


class Infinitecycle:
    '''Used to cycle over an iterable'''

    def __init__(self, create_iterable):
        self.create_iterable = create_iterable
        self.iterable = iter(create_iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterable)
        except StopIteration:
            self.iterable = iter(self.create_iterable)
            return next(self)


def get_rng_state():
    ''' returns the state of the RNGs'''
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cudarng_state = torch.cuda.get_rng_state()
        return (rng_state, cudarng_state)
    return (rng_state,)


def set_rng_state(state):
    ''' modifiy the state of the RNGs'''
    torch.set_rng_state(state[0])
    if len(state) == 2:
        torch.cuda.set_rng_state(state[1])


class EarlyStop:
    ''''Used to stop the training if the loss does not decrease during [patience] iterations'''

    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.nbest = 0
        self.niter = -1

    def step(self, y):
        self.niter += 1
        if self.best is None:
            self.nbest = 0
            self.best = y
        elif self.best < y:
            self.nbest += 1
        else:
            self.best = y
            self.nbest = 0
        return self.nbest >= self.patience


def select_batch(dataset, pin_memory, indexesiterator):
    ''' utility function for TensorDataLoader'''
    for index in indexesiterator:
        batch_index = torch.LongTensor(index)
        res = tuple(tensor.index_select(0, batch_index)
                    for tensor in dataset.tensors)
        if pin_memory and torch.cuda.is_available():
            res = tuple(x.pin_memory() for x in res)
        yield res


class TensorDataLoader:
    '''DataLoader used to iterate (efficiently!) over tuple of torch tensors'''

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
        if num_workers != 0:
            print("warning: num_workers > 0: num_workers=0 is used instead")
        sampler = RandomSampler if shuffle else SequentialSampler
        self.pin_memory = pin_memory
        self.batch_sampler = BatchSampler(sampler=sampler(
            range(len(dataset))), batch_size=batch_size, drop_last=drop_last)
        self.dataset = dataset

    def __iter__(self):
        return select_batch(self.dataset, self.pin_memory, iter(self.batch_sampler))

    def __len__(self):
        return len(self.dataset)


def str2bool(x):
    '''convert string to a boolean'''
    if x == 'False' or x == 'True':
        return eval(x)
    else:
        raise Exception(
            'str2bool: cannot convert "{} to a boolean, only "False" or "True" are accepted'.format(x))


class CsvLog:
    '''csv log writer to save all the parameters and metrics during the training'''

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.line = []

    def add2line(self, l):
        # if self.filename is not None:
        self.line += list(map(str, l))

    def writeline(self):
        # if self.filename is not None:
        prefix = "\n"
        if self.file is None:
            self.file = open(self.filename, 'w')
            prefix = ""
        self.file.write(prefix + ",".join(self.line))
        self.file.flush()
        self.line = []

    def close(self):
        if self.file is not None:# and self.filename is not None::
            self.file.close()
