import torch
from torch import nn
import torch.utils.data as data_utils
from datautils import TensorDataLoader as tensorDataLoader
import pandas as pd
import numpy as np
import pickle
import logging
from config import DATA_PATH, TRAIN, VALID, TEST, vargroupfloat, vargroupcat, choosevar
import argparse
import train
import predictionproblem
import gaussianLayer


NUM_THREAD = 0
BATCH_SIZE = 1024
COLUMN_PNLL = "pnll"


def fargs():
    parser = argparse.ArgumentParser(
        description='generates "predicted.csv" files')
    parser.add_argument('-foldermodel', default=None, type=str,
                        help="folder containing all the models")
    parser.add_argument('-fileout', default=None, type=str,
                        help="file name of the out file, typically $(DATA_PATH)/predicted.csv")
    parser.add_argument('-cpu', help='device',
                        action='store_true', default=False)
    return parser

# load the model from the pickle file 'filename'
def loadmodel(filename):
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m

# compute the predictive negative log-likelihood
def pnlGauss(e, sig2):
    return np.nanmean((e**2/sig2+np.log(sig2)+np.log(2*np.pi)) / 2, 0)


# compute the predicted mean and sigma2 for one model 'result'
def predict(device, result, test_loader):
    lmean = []
    lsig2 = []
    with torch.no_grad():
        for xy in test_loader:
            pdist = predictionproblem.predict(device, result.featuretarget, xy, result.model)
            cov = result.cmodel.outputLayer.cov(pdist)
            mean = result.cmodel.outputLayer.mean(pdist)
            lmean.append(mean.cpu())
            lsig2.append(cov.cpu())
    return np.concatenate(lmean, 0), np.concatenate(lsig2, 0)


def pnll(device, result, test_loader):
    nll = []
    with torch.no_grad():
        for xy in test_loader:
            pdist = predictionproblem.predict(device, result.featuretarget, xy, result.model)
            targ = result.featuretarget.target(xy).to(device)
            res=result.cmodel.outputLayer.normalized_nll(pdist,targ)
            nll.append(res.cpu())
    return np.concatenate(nll,0)

# compute the predicted mean and sigma2 using the ensemble of models in 'results'
def computepnll(device, y, results, test_loader):
    for result in results:
        result.model = result.model.to(device)
        result.model = result.model.eval()

    nlls = []
    for result in results:#[results[0],results[0]]:
        nll=pnll(device,result, test_loader)
#        print(nll.mean())
        nlls.append(nll)

    nlls = torch.tensor(np.array(nlls))
    mixnll = -torch.logsumexp(-nlls-np.log(nlls.shape[0]),dim=0)
    print(nlls.shape,mixnll.shape)
    print(nlls.mean(-1))
    print(mixnll.mean(),mixnll.max())
    mudf = pd.DataFrame(mixnll.numpy(), columns=[COLUMN_PNLL])
    return mudf#pd.concat([mudf, sig2df], axis=1)


def mean(device, result, test_loader):
    res = []
    with torch.no_grad():
        for xy in test_loader:
            pdist = predictionproblem.predict(device, result.featuretarget, xy, result.model)
            mean = result.cmodel.outputLayer.mean(pdist)
            res.append(mean.cpu())
    return np.concatenate(res,0)

def computemean(device,results,test_loader):
    for result in results:
        result.model = result.model.to(device)
        result.model = result.model.eval()

    nlls = []
    for result in results:#[results[0],results[0]]:
        nll=mean(device,result, test_loader)
        nlls.append(nll)

    nlls = torch.tensor(np.array(nlls))
    mixnll = torch.mean(nlls,dim=0)
    print(nlls.shape,mixnll.shape)
    print(nlls.mean(-1))
    print(mixnll.mean(),mixnll.max())
    names = ["mean_"+yvar for yvar in results[0].featuretarget.yvars]
    mudf = pd.DataFrame(mixnll.numpy(), columns=names)
    return mudf#pd.concat([mudf, sig2df], axis=1)


def computetarget(device, featuretarget, test_loader):
    res = []
    with torch.no_grad():
        for xy in test_loader:
            mean = featuretarget.target(xy)
            res.append(mean.cpu())
    names = [yvar for yvar in featuretarget.yvars]
    mudf = pd.DataFrame(np.concatenate(res,0), columns=names)
    return mudf#pd.concat([mudf, sig2df], axis=1)



# def computepnllraezra(device, y, results, test_loader):
#     for result in results:
#         result.model = result.model.to(device)
#         result.model = result.model.eval()

#     nll = torch.tensor([0.])
#     alphamodels = torch.tensor([1/len(results)])
#     for result in results:#[results[0],results[0]]:
#         tpnl=torch.tensor(pnll(device,result, test_loader))
#         nll = nll+ torch.exp(-tpnl+torch.log(alphamodels))

#     mixnll = -torch.log(nll)
# #    print(nlls.shape,mixnll.shape)
#     print(mixnll.mean(),mixnll.max())
#     mudf = pd.DataFrame(mixnll.numpy(), columns=["pnll"])
#     return mudf#pd.concat([mudf, sig2df], axis=1)

# def computepnllzraze(device, y, results, test_loader):
#     for result in results:
#         result.model = result.model.to(device)
#         result.model = result.model.eval()

#     nlls = torch.stack([torch.tensor(pnll(device,result, test_loader)) for result in results],-1)
#     print(nlls.shape)
#     logalpha = np.log(1/len(results))* torch.ones(nlls.shape)

#     mixnll = -torch.logsumexp(-nlls+logalpha,-1)
# #    print(nlls.shape,mixnll.shape)
#     print(mixnll.mean(),mixnll.max())
#     mudf = pd.DataFrame(mixnll.numpy(), columns=["pnll"])
#     return mudf#pd.concat([mudf, sig2df], axis=1)

def main():
    parser = fargs()
    args = parser.parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(args.foldermodel)
    results = [loadmodel(args.foldermodel+"/model"+str(i)+".pkl")
               for i in range(12)]
    acft_type = results[0].acft_type
    print(acft_type)
    test = train.loadcsv(acft_type, TEST)
    test_set = data_utils.TensorDataset(*results[0].featuretarget.build(test))
    test_loader = tensorDataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    y = test[results[0].featuretarget.yvars]#.reset_index(drop=True)
    df = pd.concat([computepnll(device, y, results, test_loader),computemean(device, results, test_loader),computetarget(device,results[0].featuretarget,test_loader)],sort=False,axis=1)
    df.to_csv(args.fileout, index=False)
    cov=np.cov(np.transpose(df.values[:,1:5]-df.values[:,5:]))
    cov=torch.tensor(cov)
    print(cov)
    print(torch.logdet(2*np.pi*np.e*cov)/2)
    cor=np.corrcoef(np.transpose(df.values[:,1:5]-df.values[:,5:]))
    print(cor)


if __name__ == '__main__':
    main()
