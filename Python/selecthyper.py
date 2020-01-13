import pandas as pd
import numpy as np
from config import DATA_PATH
import argparse
import os



def add_idparam(df):
    n=0
    l=[]
    oldniteration = -1
    for i in range(df.shape[0]):
        if oldniteration>df.niteration[i]:
            n+=1
        oldniteration=df.niteration[i]
        l.append(n)
    df['idparam']=np.array(l)

def load_dir(folder):
    l=[]
    for filename in os.listdir(folder):
#        print(filename)
        l.append(pd.read_csv(os.path.join(folder,filename)))
    df=pd.concat(l,ignore_index=True)
    add_idparam(df)
    return df

def fargs():
    parser = argparse.ArgumentParser(description='batchtrain predictive models.')
    parser.add_argument('-loghyperparameters',default=None,type=str)
    parser.add_argument('-loghyperparameterspcawhiten',default=None,type=str)
    parser.add_argument('-debug',default=False, action='store_true')
    parser.add_argument('-seed',default=None,type=str)
    return parser

#def build_archi(df):
#    l = [ str(int(df["archi"+str(i)])) for i in range(int(df["nlayers"]))]
#    return ",".join(l)

if __name__=='__main__':
    parser = fargs()
    args = parser.parse_args()
#    folder = os.path.join(DATA_PATH, args.model, "all/abcdeimopstz/loghyperparameters")
    df = pd.concat([load_dir(s) for s in (args.loghyperparameters,args.loghyperparameterspcawhiten) if s is not None],ignore_index=True)#folder)
    df = df.loc[df.valid.idxmin()]
    if args.debug:
        print(df)
    d = {}
#    d["-archi"] = df["archi"]
#    d["-amsgrad"] = ""
    d["-dropout2d"] = ""
#    d["-clipped"] = ""
    synonim = {"countbackprop":"niteration"}
    convert = {"niteration":int,"sum_emb":int,"batch_size":int,"nmixture":int,"freqvalid":int}
    optional = ["pca","whiten"]
    for s in ["archi","batch_size","dropout","dropout_emb","lamb","lambda_emb","countbackprop","rate","sum_emb","xvars","initrate","nmixture","freqvalid","clip"]:#,,"xnoise","ynoise"]:
        arg = synonim.get(s,s)
        d["-" + arg] = convert.get(arg,lambda x:x)(df[s])
    for s in optional:
        if df[s]:
            d["-" + s] = ""
    for seed in range(12):
        print(" ".join([k+" "+str(d[k]) for k in d]),"-seed",seed)

