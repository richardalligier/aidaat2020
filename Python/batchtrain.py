import argparse
import train
import torch
import os
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import torch

torch.set_num_threads(1)
#print("batchtrain", torch.get_num_threads())
def read_args_batch(filename):
    l=[]
    with open(filename,'r') as f:
        for line in f:
            parser=train.fargs()
            args = parser.parse_args(line.strip().split())
            l.append(args)
    return l

def check_args(l):
    for v in ["model","xvars","target","finalmodel","pca","whiten","niteration"]:
        for i,args in enumerate(l):
            if i==0:
                x=getattr(args,v)
            elif x!=getattr(args,v):
                raise Exception("check_args",v,x,getattr(args,v))

def fargs():
        parser = argparse.ArgumentParser(description='batchtrain predictive models.')
        parser.add_argument('-folderlogout',default=None,type=str)
        parser.add_argument('-foldermodelout',default=None,type=str)
        parser.add_argument('-ngpu',default=1,type=int,help='numbers gpu used')
        parser.add_argument('-nworker',default=3,type=int,help='number of workers by gpu')
        parser.add_argument('-cpu', help='device', action='store_true',default=False)
        parser.add_argument('-model',default='DH8D',type=str)
        parser.add_argument('-method',default=None,type=str)
        parser.add_argument('-batch', help='batch',default='paramlist')
        parser.add_argument('-xvars', help='batch',default="abcdeimosptz",type=str)
        parser.add_argument('-finalmodel',action='store_true',default=False)
        parser.add_argument('-pca',action='store_true',default=False)
        parser.add_argument('-whiten',action='store_true',default=False)
        parser.add_argument('-istart', help='batch',default=None,type=int)
        parser.add_argument('-iend', help='batch',default=None,type=int)
        parser.add_argument('-resume', help='batch',action='store_true',default=False)
        return parser


def worstvalue():
    '''Worst expected PNLL gaussian with cov=Diag(1,1,1,1) and mu=(0,0,0,0).'''
    n=1000
    x=torch.randn(n)
    return -4 * torch.log(torch.exp(-0.5*x**2)/np.sqrt(2*np.pi)).mean().float()

def readbest(folder,epochs):
    print(worstvalue())
    w = worstvalue()
#    best = worstvalue()*torch.ones(epochs)
    l = [pd.read_csv(os.path.join(folder,fname)) for fname in os.listdir(folder) if fname.endswith(".csv")]
    l = [torch.from_numpy(x.valid.values).float() for x in l]
    l = [ torch.cat((x,w*torch.ones(epochs-x.shape[0])),0) for x in l]
    if l==[]:
        return w*torch.ones(epochs)
    else:
        return torch.min(torch.stack(l),w)[0]

# def readbest(folder,epochs):
#     best = torch.ones(epochs)
#     l = [pd.read_csv(os.path.join(folder,fname)) for fname in os.listdir(folder) if fname.endswith(".csv")]
#     l = [torch.from_numpy(x.valid.values).float() for x in l]
#     l = [ torch.cat((x,torch.ones(epochs-x.shape[0])),0) for x in l]
#     if l==[]:
#         return torch.ones(epochs)
#     else:
#         return torch.min(torch.stack(l),0)[0]


def filterjobs(batchargs, ldataargs):
    l = [ (i,job) for i,job in enumerate(ldataargs) if (batchargs.istart is None or batchargs.istart <= i) and (batchargs.iend is None or i < batchargs.iend)]
    def getjobnumber(fname):
        return int(fname[:-len(".csv")])
    if batchargs.resume and not batchargs.finalmodel:
        alreadydone = [getjobnumber(fname) for fname in os.listdir(batchargs.folderlogout) if fname.endswith(".csv")]
        return [job for (i,job) in l if i not in alreadydone]
    else:
        return [job for (_,job) in l]

class EvenlyDispatch:
    def __init__(self,nworkers):
        self.workperworkers=torch.zeros((nworkers,),dtype=torch.int64)
        self.workperworkers.share_memory_()
        m=mp.Manager()
        self.lock=m.Lock()
    def select_gpu(self):
        '''Select the GPU'''
        with self.lock:
#            print("acquire")
            i = torch.argmin(self.workperworkers)
#            print(i)
            self.workperworkers[i]+=1
#            print("release")
        print(self.workperworkers)
        return i
    def job_done_on_gpu(self, i):
        '''One work was done on gpu i'''
        self.workperworkers[i]-=1



def train_dispatch_gpu(even,x):
    if even is None:
        train.train(*x)
    else:
        i = int(even.select_gpu())
        igpu = i % torch.cuda.device_count()
        x = (torch.device("cuda:"+str(igpu)),) + x[1:]
        print(x[0])
        with torch.cuda.device(x[0]):
            train.train(*x)
        even.job_done_on_gpu(i)

def main():
    parser = fargs()
    batchargs = parser.parse_args()
    device = torch.device("cpu" if batchargs.cpu else "cuda")
    even = None if batchargs.ngpu<=1 else EvenlyDispatch(batchargs.ngpu)
    largs = read_args_batch(batchargs.batch)
    check_args(largs)
    for i,args in enumerate(largs):
        args.xvars = "".join(sorted(batchargs.xvars))
        args.model = batchargs.model
        args.method = batchargs.method
        if batchargs.pca:
            args.pca = batchargs.pca
        if batchargs.whiten:
            args.whiten = batchargs.whiten
        args.cpu = batchargs.cpu
        args.filelog = "/dev/null" if batchargs.folderlogout is None else batchargs.folderlogout+"/"+str(i)+".csv"
        args.finalmodel = batchargs.finalmodel
        args.save_model = None if batchargs.foldermodelout is None else batchargs.foldermodelout+"/model"+str(i)+".pkl"
    best = None
    if not batchargs.finalmodel:
        best = readbest(batchargs.folderlogout,largs[0].niteration)#torch.ones(largs[0].epochs)
        best.share_memory_()
    data = train.load_data(largs[0],share_memory=True)
    ldataargs = [(device,data,args,best) for args in largs]
    ldataargs = filterjobs(batchargs, ldataargs)
    ldataargs = [(even,x) for x in ldataargs]
    print("jobs to do:",len(ldataargs))
    with mp.Pool(batchargs.ngpu * batchargs.nworker) as p:
        p.starmap(train_dispatch_gpu, ldataargs)

if __name__ == '__main__':
    main()
