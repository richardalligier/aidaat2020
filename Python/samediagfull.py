import torch
from torch import nn
import torch.utils.data as data_utils
from datautils import TensorDataLoader as tensorDataLoader
import pandas as pd
import numpy as np
import pickle
import logging
from config import DATA_PATH, TRAIN, VALID, TEST, GBM_RES
import argparse
import train
#import predictionproblem
#import gaussianLayer
import sklearn
import test
import os

NUM_THREAD = 0
BATCH_SIZE = 1024


def fargs():
    parser = argparse.ArgumentParser(
        description='generates "predicted.csv" files')
    # parser.add_argument('-foldermodel', default=None, type=str,
    #                     help="folder containing all the models")
    parser.add_argument('-method', default=None, type=str,
                        help="method used")
    parser.add_argument('-model', default=None, type=str,
                        help="aircraft type")
    parser.add_argument('-fileout', default=None, type=str,
                        help="file name of the out file, typically $(DATA_PATH)/predicted.csv")
    # parser.add_argument('-cpu', help='device',
    #                     action='store_true', default=False)
    return parser


class SimpleGMM:
    def __init__(self, k):
        self.gmm = sklearn.mixture.GaussianMixture(k,verbose=2,verbose_interval=1)#,tol=1e-15)

    def fit(self, yscale):
        self.gmm.fit(yscale)
#        print(self.gmm.covariances_)
    def predict(self, data):
        nll = -self.gmm.score_samples(data)
        print("mean nll",nll.mean())
        return pd.DataFrame(nll, columns= [test.COLUMN_PNLL])

        # print(et.shape)
        # precisiont = torch.tensor(self.precision).float().expand(et.shape[0],-1,-1)
        # # print(self.precision.shape,e.shape)
        # # print(torch.bmm(precisiont,et).shape)
        # ele = torch.bmm(torch.transpose(et,-1,-2),torch.bmm(precisiont,et))
        # print(ele)
        #np.dot(np.transpose(e),np.dot(self.precision,e))

#        pass



class SimpleModel:
    def __init__(self,full=False):
        self.full = full

    def fit(self, yscale):
        self.cov = np.cov(yscale,rowvar=False)
        self.mean = np.mean(yscale,0)
        if not self.full:
            self.cov = np.diag(np.diag(self.cov))
        self.precision = np.linalg.inv(self.cov)
        self.m = yscale.shape[1]
        print("used cov",self.cov)
        print("used mean",self.mean)
    def predict(self, data):
        e = data - self.mean
        pe = np.matmul(self.precision, np.transpose(e))
        epe = np.sum(e*np.transpose(pe),1)
        logd = -np.log(np.linalg.det(self.precision)).sum()
        nll = (epe + logd + self.m * np.log(2*np.pi)) *0.5
        # print(res)
        # print(epe[:10])
        print("mean nll",nll.mean())
        # et = torch.tensor(e).float().unsqueeze(-1)
        # print(et.shape)
        # precisiont = torch.tensor(self.precision).float().expand(et.shape[0],-1,-1)
        # # print(self.precision.shape,e.shape)
        # # print(torch.bmm(precisiont,et).shape)
        # ele = torch.bmm(torch.transpose(et,-1,-2),torch.bmm(precisiont,et))
        # print(ele[:10])
        return pd.DataFrame(nll, columns= [test.COLUMN_PNLL])
        #np.dot(np.transpose(e),np.dot(self.precision,e))

#        pass



def loadgbm(model):
    ''' load values predicted by GBM (TRC2018) '''
    bf = train.loadcsv(model, TEST,drop_unknown=False,drop_not_in_time_frame=False)
    # fullfilename = DATA_PATH+"/foldedtrajs/{}_test.csv.xz".format(model)
    # if os.path.exists(fullfilename+".pkl"):
    #     bf = pickle.load(open(fullfilename+".pkl", "rb"))
    # else:
    #     bf = pd.read_csv(fullfilename, usecols=[
    #                      "maxtimestep", "timestep", "mseEnergyRateFutur"]+ALLV)
    #     pickle.dump(bf, open(fullfilename+".pkl", "wb"), protocol=4)
    # bf = bf.query('maxtimestep>=timestep+600').query("timestep>=135").reset_index(drop=True)
    bf = bf.query("timestep>=135").query('maxtimestep>=timestep+150').reset_index(drop=True)
    print(bf.shape)
    for v in train.ALLVARS:
        gbmdf = pd.read_csv(
            GBM_RES + "/{}/{}/abdemopst/gbmpredicted.csv".format(model, v))
        print(v,gbmdf.shape)
        bf["pred"+v] = gbmdf["pred"+v].values
    bf = bf.query('maxtimestep>=timestep+600').query("timestep>=135")
    for v in train.ALLVARS:
        bf = bf.query(v+"=="+v)
    for v in train.ALLVARS:
        print(np.sqrt(np.mean((bf[v]-bf["pred"+v])**2)))
    print(bf.shape)
    return bf.reset_index(drop=True)


def main():
    parser = fargs()
    args = parser.parse_args()
    # device = torch.device(
    #     "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
#    acft_type = args.model

    scaler = sklearn.preprocessing.StandardScaler()
    data_train = pd.concat([train.loadcsv(args.model,TRAIN),train.loadcsv(args.model,VALID)])
    yscaledtrain = scaler.fit_transform(data_train[train.ALLVARS].values)
    data_test = train.loadcsv(args.model, TEST)
    yscaledtest=scaler.transform(data_test[train.ALLVARS].values)
    if "gb" in args.method:
        pred = loadgbm(args.model)
    if args.method == "diag":
        model = SimpleModel()
    elif args.method == "full":
        model = SimpleModel(True)
    elif args.method == "gmm":
        model = SimpleGMM(10)
    elif args.method == "gbdiag":
        model = SimpleModel()
        yscaledtrain = yscaledtest-scaler.transform(pred[["pred"+v for v in train.ALLVARS]].values)
        yscaledtest = yscaledtrain
    elif args.method == "gbfull":
        model = SimpleModel(True)
        yscaledtrain = yscaledtest-scaler.transform(pred[["pred"+v for v in train.ALLVARS]].values)
        yscaledtest = yscaledtrain
    else:
        raise Exception
    model.fit(yscaledtrain)
    df = model.predict(yscaledtest)
    if args.fileout is not None:
        df.to_csv(args.fileout, index=False)

if __name__ == '__main__':
    main()
