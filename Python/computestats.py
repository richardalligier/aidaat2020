import numpy as np
import pandas as pd
import os
import pickle
from config import DATA_PATH, GBM_RES, TEST, TABLE_FOLDER
import train
import test

METHODNAME= {
    "samediag":" $\mathrm{Diag}$",
    "samefull":"$\mathrm{Full}$",
    "gbdiag":"${\mathrm{GB}(x)}_{\mathrm{Diag}}$",
    "gbfull":"${\mathrm{GB}(x)}_{\mathrm{Full}}$",
    "diag":"$\mathrm{Diag}(x)$",
    "full":"$\mathrm{Full}(x)$",
    "gmm":"$\mathrm{GMM}(x)$",
}

METHODSPACE= {
    "samediag":0.4,
    "samefull":0.4,
    "gbdiag":0.9,
    "gbfull":0.9,
    "diag":0.6,
    "full":0.6,
    "gmm":0.6,
}

def getpathpred(model, variables="abcdeimopstz"):
    '''build the path to the predicted.csv file'''
    return os.path.join(DATA_PATH, model, 'aidaat2020all', variables, "predicted.csv")


def loaddf(model, methods, stat):
    l = [model]
    for method in methods:
        try:
            df = pd.read_csv(getpathpred(model, variables=method))
            l.append("{:.2f}".format(stat(df[test.COLUMN_PNLL].values)))
        except FileNotFoundError:
            l.append("None")
    return l


def table(dmodel, methods, fileout):
    '''build a table'''
    l = []
    df = pd.DataFrame(data=dmodel, columns=["model"]+[METHODNAME[m] for m in methods])
    column_format = "r"+"".join(["p{"+str(METHODSPACE[m])+"cm}" for m in methods])
    for i in range(1,len(methods)):
        im = np.mean(df.values[:,i].astype(np.float))
        im1 = np.mean(df.values[:,i+1].astype(np.float))
        print(i,i+1,im,im1,im-im1)#,np.mean(df.values[:,i].astype(np.float)-df.values[:,i+1].astype(np.float)))
    with open(fileout, 'w') as f:
        f.write(df.to_latex(escape=False, index=False,column_format=column_format))


def dotables():
    lmodel = sorted(['A320','E190','E195','DH8D','B737','CRJ9','A332','B77W','A319','A321','B738'])
    methods = ["samediag","samefull","gbdiag","gbfull","diag","full","gmm"]
#    methods = ["samediag","samefull","gbdiag","gbfull","gmm"]
    dmodel = [loaddf(model,methods,np.mean) for model in lmodel]
    table(dmodel, methods, TABLE_FOLDER+"/tablemean.tex")
    def sigmaeq(v):
        meannll = np.mean(v)
        return np.exp(meannll/2)/(2*np.pi*np.e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-which', type=str)
    args = parser.parse_args()
    lwhich = (("tables", dotables), )
    done = False
    for (s, f) in lwhich:
        if args.which == s:
            f()
            done = True
    if not done:
        raise Exception(
            "-which accept only {}".format(tuple(s for s, _ in lwhich)))
