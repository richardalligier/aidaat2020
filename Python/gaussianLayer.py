import torch
import numpy as np
from config import DATA_PATH, TEST

from block import Softplusmin

import torch.nn.functional as F

import sklearn
import sklearn.mixture

PI = np.pi

MINSIG2 = 1e-6
INV_SOFTPLUS1 = np.log(np.exp(1)-1)

def torchtril(x,diagonal=0):
    ok = (x.shape[0] != 1) or len(shape) != 4
    assert ok # https://github.com/pytorch/pytorch/issues/22581
    return torch.tril(x,diagonal)


def tuple_to(v, device):
    return tuple(x.to(device) if not isinstance(x,tuple) else tuple_to(x,device) for x in v)


def target_numpy_from_train_loader(target, train_loader):
    l = []
    for xy in train_loader:
        l.append(target(xy).cpu().numpy())
    return np.concatenate(l)

class MuSigmaDiag(torch.nn.Module):
    def __init__(self, size_in, size_y):
        super().__init__()
        print("size_in, size_y",size_in, size_y)
        self.mu = torch.nn.Linear(size_in,size_y)
        self.sig2 = torch.nn.Linear(size_in,size_y)
        self.clamp = Softplusmin(MINSIG2, alpha=1)#clamp
    def forward(self, x):
        mu = self.mu(x)
        sig2 = self.clamp(self.sig2(x))
        return (mu, sig2)
    # def loss(self, pdist, yobs):
    #     mu ,sig2 = pdist
    #     return ((mu-yobs)**2/sig2 + torch.log(sig2)).sum(-1)
    def normalized_nll(self, pdist, yobs):
        mu,sig2=pdist
        return ((mu-yobs)**2/sig2 + torch.log(sig2)+np.log(2*PI)).sum(-1)*0.5
    def cov(self, pdist):
        return torch.diag_embed(pdist[1])
    def init_weights(self):
        ps = list(self.mu.parameters())+list(self.sig2.parameters())
        for x in ps:
            torch.nn.init.constant_(x,0.)
    def mean(self, pdist):
        return pdist[0]



class MuSigmaFull(torch.nn.Module):
    def __init__(self, size_in, size_y):
        super().__init__()
        print("size_in, size_y",size_in, size_y)
        self.mu = torch.nn.Linear(size_in,size_y)
        self.tril = torch.nn.Linear(size_in, size_y*size_y)
        self.diag = torch.nn.Linear(size_in, size_y)
        self.clamp = Softplusmin(MINSIG2, alpha=1)#clamp
        self.size_y = size_y
#        self.debug = True

    def forward(self, x):
        mu = self.mu(x)
        diag = 1/torch.sqrt(self.clamp(self.diag(x)))#torch.clamp(self.clamp(self.diag(x)),max=1e2)
#        diag = self.clamp(self.diag(x))#torch.clamp(self.clamp(self.diag(x)),max=1e2)
        # if self.debug:
        #     print("diag", diag)
        #     self.debug = False
        tril = torchtril(self.tril(x).reshape(-1,self.size_y,self.size_y),diagonal=-1)
#        print(tril,tril.shape)
        l = torch.diag_embed(diag) + tril
        precision =  torch.bmm(l,torch.transpose(l,-1,-2))
#        print(precision,precision.shape,diag.shape)
        return mu, (diag, precision)

    # def loss(self, pdist, yobs):
    #     mu ,sig2 = pdist
    #     (diag, precision) = sig2
    #     e = (yobs - mu).unsqueeze(-1)
    #     res =  torch.bmm(torch.transpose(e,-1,-2),torch.bmm(precision,e)).squeeze(-1).squeeze(-1)#.mean()
    #     logd = -2*torch.log(diag).sum(-1)#.mean()TODO
    #     return (res+logd)#/self.size_y

    def normalized_nll(self,pdist,yobs):
        mu,sig2=pdist
        (diag, precision) = sig2
        e = (yobs - mu).unsqueeze(-1)#why unsqueeze(-1)???
        res =  torch.bmm(torch.transpose(e,-1,-2),torch.bmm(precision,e)).squeeze(-1).squeeze(-1)#.mean()
        logd = -2*torch.log(diag).sum(-1) #=torch.logdet(precision)
        return (res + logd + self.size_y*np.log(2*PI))*0.5

    def cov(self, pdist):
        return pdist[1][1].inverse()#torch.diag_embed(sig2[0])
    def mean(self, pdist):
        return pdist[0]
    def init_weights(self):
        ps = list(self.mu.parameters())+list(self.diag.parameters())+list(self.tril.parameters())
        for x in ps:
            torch.nn.init.constant_(x,0.)
    def _init_weights_mixture(self):
        ps = list(self.diag.parameters())+list(self.tril.parameters())
        a,b = self.mu.parameters()
        torch.nn.init.constant_(a,0.)
        torch.nn.init.normal_(b,std=0.5)
#        torch.nn.init.constant_(b,0.)
        for x in ps:
            torch.nn.init.constant_(x,0.)

class Mixture(torch.nn.Module):
    def __init__(self, size_in, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.choicemixture = torch.nn.Linear(size_in,len(models))
        self.logsoftmax = torch.nn.LogSoftmax()

    def forward(self,x):
        logalpha = self.logsoftmax(self.choicemixture(x))
        return logalpha, tuple(model(x) for model in self.models)

    def normalized_nll(self, pdist, yobs):
        logalpha, pdists = pdist
        nlls = torch.stack([model.normalized_nll(pdisti,yobs) for (model,pdisti) in zip(self.models,pdists)],-1)
#        print(nlls.sum())
        # print(logalpha[:10])
        # print((-nlls)[:10])
        # print((-nlls+logalpha)[:10])
        # print(-torch.logsumexp(-nlls+logalpha,-1)[:10])
        return -torch.logsumexp(-nlls+logalpha,-1)
    # def loss(self, pdist, yobs):
    #     return self.normalized_nll(pdist,yobs)
    def mean(self, pdist):
        logalpha, pdists = pdist
        means = torch.stack([model.mean(pdisti) for (model,pdisti) in zip(self.models,pdists)],-2)
        # print(torch.exp(logalpha))
        return torch.sum(torch.exp(logalpha.unsqueeze(-1))*means,-2)
    def init_weights(self):
        # for model in self.models:
        #     model.init_weights()
        ps = list(self.choicemixture.parameters())
        for x in ps:
            torch.nn.init.constant_(x,0.)
        for m in self.models:
            m._init_weights_mixture()

    def cov(self, pdist):
        for i,model in enumerate(self.models):
            print(model.cov(pdist[1][i])[0])


def transform_mixture(x):
    return F.log_softmax(x,-1)

def inv_transform_mixture(alpha):# alpha! not logalpha !!!!
    return torch.log(alpha)

def torchinv_softplus(y):
    return torch.log(torch.exp(y)-1)


def inv_softplus(y):
    return np.log(np.exp(y)-1)

def transform_d(x):
    xc = x + INV_SOFTPLUS1
    return 1/torch.sqrt(MINSIG2+F.softplus(xc))

def inv_transform_d(y):
    softplusxc = 1/y**2-MINSIG2
    xc = torchinv_softplus(softplusxc)
    return xc - INV_SOFTPLUS1

#GMM = namedtuple('GMM',['logalpha','mu','l','d'])
class GaussianMixture(torch.nn.Module):
    def __init__(self, size_in, K, size_out):
        super().__init__()
        self.K = K
        self.size_out = size_out
        self.choicemixture = torch.nn.Linear(size_in, K)
        self.mu = torch.nn.Linear(size_in, K * size_out)
        self.l = torch.nn.Linear(size_in, K * size_out * size_out)
        self.d = torch.nn.Linear(size_in, K * size_out)
#        self.clamp = Softplusmin(MINSIG2, alpha=1)#clamp



    def forward(self,x):
        logalpha = transform_mixture(self.choicemixture(x))
        mu = self.mu(x)
        l = self.l(x)
        d = transform_d(self.d(x))
        # d = self.d(x)
        # d = 1/torch.sqrt(self.clamp(d))
        return (logalpha, mu, l, d)

    def _get_precision(self, pdist):
        (_,_,l,d) = pdist
        l =l.reshape(-1, self.K, self.size_out, self.size_out)#.contiguous()
        l = torch.tril(l, diagonal=-1)
        d = d.reshape(-1, self.K, self.size_out)
        diag = torch.diag_embed(d)#.reshape(-1, self.K, self.size_out))
        t = l + diag
        lamb = torch.matmul(t,torch.transpose(t,-1,-2))
        lamb = lamb.reshape(-1,self.size_out,self.size_out)
        return lamb, d

    def normalized_nll(self, pdist, yobs):
        (logalpha,mu,_,_) = pdist
        lamb, d = self._get_precision(pdist)
#        print(l.shape)
        yobs = yobs.unsqueeze(-2)
        mu = mu.reshape(-1,self.K, self.size_out)
#        print("yobs.shape,mu.shape",yobs.shape,mu.shape)
        e = (yobs - mu).reshape(-1,self.size_out,1)
#        print("e.shape",e.shape,lamb.shape)
        res =  torch.bmm(torch.transpose(e,-1,-2),torch.bmm(lamb,e))#torch.matmul(torch.transpose(e,-1,-2),torch.matmul(lamb,e))#.squeeze(-1).squeeze(-1)#.mean()
#        print("res.shape",res.shape)
        res = res.reshape(-1,self.K)
#        print("res.shape",res.shape)
        logd = torch.log(d).sum(-1) #=torch.logdet(precision)
        assert res.shape == logd.shape
        nlls = (res + self.size_out * np.log(2*PI)-2*logd) * 0.5
#        print(nlls.sum())
        assert nlls.shape == logalpha.shape
        return -torch.logsumexp(-nlls + logalpha, -1)
#        return (res + logd + self.size_y*numpy.log(2*PI))*0.5

#        nlls = torch.stack([model.normalized_nll(pdisti,yobs) for (model,pdisti) in zip(self.models,pdists)],-1)
        # print(logalpha[:10])
        # print((-nlls)[:10])
        # print((-nlls+logalpha)[:10])
        # print(-torch.logsumexp(-nlls+logalpha,-1)[:10])
#        return -torch.logsumexp(-nlls+logalpha,-1)

    def mean(self, pdist):
        (logalpha,mu,l,d) = pdist
        mu = mu.reshape(-1,self.K, self.size_out)
        mean=torch.mean(mu,-2)
#        print("mean.shape",mean.shape)
        return mean#torch.mean(mu,-2)

    def init_weights(self):
        ps = list(self.choicemixture.parameters())
        for x in ps:
            torch.nn.init.constant_(x,0.)
        for x in self.l.parameters():
            torch.nn.init.constant_(x,0.)
        a,b=self.d.parameters()
        torch.nn.init.constant_(a,0.)
        torch.nn.init.constant_(b,0.)
#        torch.nn.init.normal_(b,std=0.5)
        a,b = self.mu.parameters()
        torch.nn.init.constant_(a,0.)
        torch.nn.init.normal_(b,std=0.5)
        # for m in self.models:
        #     m._init_weights_mixture()
    def init_weights_EM(self, y):#target, train_loader):
#        y = target_numpy_from_train_loader(target, train_loader)
        gmm = sklearn.mixture.GaussianMixture(self.K)
        gmm.fit(y)
        a,b = self.mu.parameters()
        torch.nn.init.constant_(a,0.)
        # print(gmm.means_)
        # print(b.shape)
        with torch.no_grad():
            b.copy_(torch.tensor(gmm.means_,requires_grad=False).reshape(-1))
        # print(b)
        # print(b.reshape(self.K,self.size_out))
#        print(gmm.precisions_cholesky_)
        pchol = torch.cholesky(torch.tensor(gmm.precisions_))
        # print(pchol)
        eml = torchtril(pchol,diagonal=-1)#.contiguous()
        a,b = self.l.parameters()
        torch.nn.init.constant_(a,0.)
        # print(b.shape)
        with torch.no_grad():
            b.copy_(eml.reshape(-1))
        # print(b)
        # print(b.reshape(self.K,self.size_out,self.size_out))
        a,b = self.d.parameters()
        emd = torch.diagonal(pchol,dim1=-2,dim2=-1)
        # print(emd)
        torch.nn.init.constant_(a,0.)
        # print(b.shape)
        with torch.no_grad():
            b.copy_(inv_transform_d(emd.reshape(-1)))
        # print(b)
        # print(transform_d(b.reshape(self.K,self.size_out)))

        a,b = self.choicemixture.parameters()
        emalpha = torch.tensor(gmm.weights_)
        # print(emalpha)
        # print(F.softmax(inv_transform_mixture(emalpha)))
        torch.nn.init.constant_(a,0.)
        # print(b.shape)
        with torch.no_grad():
            b.copy_(emalpha)
        # print(b)
        # print(b.reshape(self.K))
#        raise Exception

    def cov(self, pdist):
        lamb, d = self._get_precision(pdist)
        return torch.inverse(lamb[0]),1/d[0]

def random_cov(dim):
    l = torchtril(torch.randn((dim,dim)),diagonal=-1)
    d = torch.diag_embed(torch.exp(torch.randn(dim)))
    l = l+d
    return torch.mm(l,torch.transpose(l,-1,-2))

# def test():
#     from block import Softplusmin
#     torch.manual_seed(1)
#     clamp = Softplusmin(1e-6)#torch.nn.Softplus()
#     dim = 4
#     xdim = 1
#     N = 1000

#     K = 2
#     cov = random_cov(dim)
#     print("best",0.5*torch.logdet(2*PI*np.e*cov))
#     m = GaussianMixture(xdim,K,dim)
# #    m = MuSigmaDiag(xdim, dim, clamp)
# #    m = Mixture(xdim,[MuSigmaFull(xdim, dim) for i in range(K)])
#     m.init_weights_EM()
#     print(cov)
#     g = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim),cov)
#     print(list(m.parameters()))
# #    raise Exception
#     optim = torch.optim.Adam(m.parameters(),lr=1e-1,weight_decay=0)
#     scheduler = torch.optim.lr_scheduler.StepLR(optim,50,gamma=0.1)
#     niter = 100
#     print(list(m.parameters()))
#     for _ in range(niter):
#         optim.zero_grad()
#         yobs = torch.stack([g.sample() for _ in range(N)],0)
#         x = torch.zeros((N,xdim))
#         pdist = m(x)
#         loss = m.normalized_nll(pdist, yobs).mean()
#         print(loss)
#         loss.backward()
#         optim.step()
# #        scheduler.step()
#     p=list(m.parameters())
#     print(p,clamp(p[-1]))
#     print("predicted mu",m.mean(pdist)[0])
#     print(torch.exp(pdist[0]))
#     print(cov)
#     print(m.normalized_nll(pdist,yobs).mean())
#     print(m.cov(pdist))


def empcov(ys):
    mu = torch.mean(ys,0)
    e= ys - mu
    print(torch.mm(e.t(),e)/e.shape[0])



def testDH8D():
    from block import Softplusmin
    import pandas as pd
    import train
#    import matplotlib.pyplot as plt
    import sklearn
    import sklearn.mixture
    dim = 4
    xdim = 1
    N = 1000
    K = 1
    torch.manual_seed(0)
    acft = "DH8D"
    ts = train.loadcsv(acft, TEST,dropunknown=False)
    pred = pd.read_csv(DATA_PATH + "/{}/atm2019all/abcdeimopstz/predicted.csv".format(acft))
    ALLVAR = ["massFutur", "target_cas1", "target_cas2", "target_Mach"]
    df = ts[ALLVAR]
    print(df.shape,pred.shape)
    df = pd.concat([df,pred],axis=1)
    for xvar in ALLVAR:
        df = df.query(xvar + "==" + xvar).reset_index(drop=True)
    print(list(df))
    y = torch.tensor(df[ALLVAR].values)
    # ys = (y-torch.mean(y,0))/torch.std(y,0)
    s = torch.std(y, 0)
#    ys = y-torch.mean(y,0)
    ys = y-torch.tensor(df[["pred"+x for x in ALLVAR]].values,dtype=torch.float)
    ys = ys / s
    print(ys[:10])
    # ys = ys/torch.sqrt(torch.tensor(df[["sig2"+x for x in ALLVAR]].values,dtype=torch.float))
    # print(ys[:10])


    # plt.scatter(ys[:,2].numpy(),ys[:,3].numpy(),alpha=0.01)
    # plt.show()
    # plt.scatter(ys[:,0].numpy(),ys[:,3].numpy(),alpha=0.01)
    # plt.show()
    gmm = sklearn.mixture.GaussianMixture(K)
    gmm.fit(ys)
    print("gmm.score(ys)",gmm.score(ys))
    # print(gmm.weights_)
    # print(gmm.means_)
    # print(gmm.precisions_)
    # print(gmm.precisions_cholesky_)
    # print(np.dot(gmm.precisions_cholesky_[0],np.transpose(gmm.precisions_cholesky_[0])))
    # print(np.cov(np.transpose(ys.numpy())))
    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(ys.shape[0])), batch_size=N, drop_last=True)
#    m = MuSigmaFull(xdim, dim)
#    m = MuSigmaFull(xdim, dim)
    m = GaussianMixture(xdim, K, dim)
    m.init_weights()#_EM(ys)
#    m = Mixture(xdim,[MuSigmaDiag(xdim, dim, clamp) for i in range(K)])
#    g = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(dim),cov)
    optim = torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,50,gamma=0.1)
    niter = 50
    print(list(m.parameters()))
    for _ in range(niter):
        for ind in sampler:
            ind = torch.LongTensor(ind)
            yobs = torch.index_select(ys,0,ind)
            optim.zero_grad()
            x = torch.zeros((N,xdim))
            pdist = m(x)
            # a,b=m.d.parameters()
            # c,d=m.l.parameters()
            loss = m.normalized_nll(pdist, yobs).mean()#+10*(b**2).sum()#+10*(d**2).sum()
            print(loss)
            loss.backward()
            optim.step()
#        scheduler.step()
    p=list(m.parameters())
    pdist = m(x)
#    print("predicted mu",mu)
    print("cov")
    print(m.cov(pdist))
    print(np.cov(np.transpose(ys.numpy())))
    print(torch.sqrt(torch.mean(ys**2,0)))
    e = df.massFutur-df.predmassFutur
    print(e.describe())
    print(df.massFutur.describe())
    print(pred.shape,ts.shape,df.shape)
    # print(m.cov(sig2))
    # print(m.normalized_nll(mu,sig2,yobs).mean())
    # print(empcov(ys))

if __name__=="__main__":
    testDH8D()
