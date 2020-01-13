import torch
import numpy as np


def build_FCNWithoutOutput(archi, dropout, nin, useresidual=False):
    '''Builds a fully connected feed-forward neural network'''
    print(archi)
    nonlinear = torch.nn.SELU#LeakyReLU#torch.nn.CELU
    nmaxout = archi[0]

    def block(i, nin, nout):
        l = [torch.nn.Linear(nin, nout), nonlinear()]
        if dropout != 0.:
            l.append(torch.nn.Dropout(min(dropout/nmaxout*nout, 0.9)))
        return l

    def blockwithres(i, nin, nout):
        b = block(i, nin, nout)
        if useresidual and nin == nout:
            return [Resblock(torch.nn.Sequential(*b))]
        else:
            return b
    l = []
    archi = [nin] + archi
    if archi[-1] != 0:
        for i in range(len(archi) - 1):
            l += blockwithres(i, archi[i], archi[i + 1])
#        l.append(torch.nn.Linear(archi[-1], nout))
    # else:
    #     l.append(torch.nn.Linear(nin, nout))
    return l


class ApplySlice_(torch.nn.Module):
    '''utility class used to define other Classes like Concat, Map, ...'''

    def __init__(self, start=None, end=None):
        super(ApplySlice_, self).__init__()
        self.start = start
        self.end = end

    def applyslice(self, x):
        raise NotImplemented

    def extra_repr(self):
        return 'start={}, end={}'.format(self.start, self.end)

    def forward(self, x):
        start = self.start
        end = self.end
        mapped = self.applyslice(x[start:end])
        if start is None:
            if end is None:
                return mapped
            else:
                return mapped+x[end:]
        else:
            if end is None:
                return x[:start]+mapped
            else:
                return x[:start]+mapped+x[end:]


class Concat(ApplySlice_):
    '''concatenate the component of a tuple'''

    def __init__(self, start=None, end=None):
        super(Concat, self).__init__(start, end)

    def applyslice(self, x):
        res = torch.cat(x, -1)
        if self.start is None and self.end is None:
            return res
        else:
            return (res,)


class Map(ApplySlice_):
    '''Apply the function fi to the component xi of a tuple'''

    def __init__(self, modules, start=None, end=None):
        super(Map, self).__init__(start, end)
        self.in_modules = torch.nn.ModuleList(modules)

    def applyslice(self, x):
        return tuple(m(xi) for xi, m in zip(x, self.in_modules))
    def __iter__(self):
        return iter(self.in_modules)


class Sum(ApplySlice_):
    '''Sum the components of a tuple'''

    def __init__(self, start=None, end=None):
        super(Sum, self).__init__(start, end)

    def applyslice(self, x):
        res = torch.sum(torch.stack(x, -1), -1)
        if self.start is None and self.end is None:
            return res
        else:
            return (res,)


class Mean(ApplySlice_):
    '''Compute the mean the components of a tuple'''

    def __init__(self, start=None, end=None):
        super(Mean, self).__init__(start, end)

    def applyslice(self, x):
        res = torch.sum(torch.stack(x, -1), -1)/len(x)
        if self.start is None and self.end is None:
            return res
        else:
            return (res,)


class Debug(torch.nn.Module):
    '''Class used to debug, especially the dead ReLUs unit'''

    def forward(self, x):
#        print(x.shape)
        if True:#not self.training:
            print(x.shape)
            activations = x.std(0)
            print(activations.mean(),activations.std())
#            print(torch.sum((x <= 0).int(), 0).float()/x.shape[0])
        return x


class Unsqueeze(torch.nn.Module):
    '''Module unsqueezing a tensor'''

    def forward(self, x):
        return x.unsqueeze(0)


class Squeeze(torch.nn.Module):
    '''Module squeezing a tensor'''

    def forward(self, x):
        return x.squeeze(0)


class Split(torch.nn.Module):
    '''build a tuple from a tensor'''

    def __init__(self, i):
        super(Split, self).__init__()
        self.i = i

    def forward(self, x):
        return (x[:, :self.i], x[:, self.i:])

    def extra_repr(self):
        return 'i={}'.format(self.i)


class Softplusmin(torch.nn.Module):
    '''used to obtain a positive output no matter the input'''

    def __init__(self, minlim, beta=1., alpha=1.):
        super().__init__()
        self.minlim = minlim
        self.beta = beta
        self.alpha = alpha
        self.c = np.log(np.exp(self.beta)-1)/self.beta-self.minlim
        self.elu = torch.nn.Softplus(beta=self.beta)

    def forward(self, x):
        return self.minlim+self.elu((self.alpha*x+self.c))

    def extra_repr(self):
        return 'min={}'.format(self.minlim)


class Resblock(torch.nn.Module):
    '''Module used for ResNet'''

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class GaussianNoise(torch.nn.Module):
    '''Module used for ResNet'''

    def __init__(self, std):
        super().__init__()
        self.std = std
        self.oldtraining=self.training

    def forward(self, x):
        if self.oldtraining!=self.training:
            print("self.training",self.training)
        self.oldtraining=self.training
        if self.training:
            return x + self.std*torch.randn_like(x)
        else:
            return x


class Count:
    def __init__(self):
        self.c = 0
    def incr(self):
        self.c += 1
