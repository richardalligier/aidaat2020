from config import DATA_PATH, TRAIN, VALID, TEST, vargroupfloat, vargroupcat, choosevar
from datautils import EarlyStop, CsvLog, Infinitecycle, TensorDataLoader as tensorDataLoader
from torch import nn
import torch
import time
import gc
import numpy as np
import pandas as pd
import pickle
import os
import argparse

from loss import gen_mtlLoss, gen_nll
import predictionproblem

torch.set_num_threads(1)


ALLVARS = ["massFutur", "target_cas1", "target_cas2", "target_Mach"]
BATCH_SIZE_VALID = 1024*8
MAX_FREQVALID = 15000

def yvars(args):
    ''' returns a list of target variables'''
    return ALLVARS if args.target == "all" else [args.target]


class TrainResult:
    '''Class used to embedd the saved model'''
    def __init__(self, acft_type, featuretarget, cmodel):
        self.acft_type = acft_type
        self.cmodel = cmodel
        self.featuretarget = featuretarget
        self.model = cmodel.model


def fargs():
    parents = [predictionproblem.fargs()]
    parser = argparse.ArgumentParser(
        description='train a predictive model.', parents=parents)
    parser.add_argument('-model', help='aircraft model', default="DH8D")

    parser.add_argument('-cpu', help='device',
                        action='store_true', default=False)
#    parser.add_argument('-cuda', help='device', default=0, type=int)
    parser.add_argument('-niteration', help='niteration',
                        default=5000, type=int)
    parser.add_argument('-rate', help='learning rate decay',
                        default=1., type=float)
    parser.add_argument('-target', help='target', default="all")
    parser.add_argument('-seed', help='seed', default=0, type=int)
    parser.add_argument('-filelog', help='filelog', default="log", type=str)
    parser.add_argument('-finalmodel', action='store_true', default=False)
    parser.add_argument('-tolbest', default=0.3, type=float)
    parser.add_argument('-thresholdlrstop', default=1e-50, type=float)
    parser.add_argument('-ynoise', default=None, type=float)
    parser.add_argument('-xvars', default="abcdeimopstz", type=str)
    parser.add_argument('-save_model', default=None, type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument("-freqvalid", default=None, type=int)
    parser.add_argument("-factorlowertriangular", default=1e-5, type=float)
    parser.add_argument("-modecyclic", default='triangular', type=str)
    return parser


def loadcsv(model, setname, drop_unknown=True,drop_not_in_time_frame=True):
    ''' load csv trajectories file, filters it and returns a pandas dataframe'''
    filename = "{}_{}.csv.xz".format(model, setname)
    fullfilename = os.path.join(DATA_PATH, "foldedtrajs", filename)
    if os.path.exists(fullfilename+".pkl"):
        df = pickle.load(open(fullfilename+".pkl", "rb"))
    else:
        usecols = choosevar(vargroupfloat, "".join(vargroupfloat.keys()))+choosevar(vargroupcat, "".join(vargroupcat.keys()))+[
            "massFutur", "target_Mach", "target_cas1", "target_cas2", "maxtimestep", "timestep", "segment", "mseEnergyRateFutur"]
        df = predictionproblem.loadcsv(fullfilename, usecols)
        pickle.dump(df, open(fullfilename+".pkl", "wb"), protocol=4)
    if drop_not_in_time_frame:
        df = df.query('maxtimestep>=timestep+600').query("timestep>=135")
    if drop_unknown:
        for x in ["massFutur", "target_Mach", "target_cas1", "target_cas2"]:
            df = df.query(x+"=="+x)
    df = df.reset_index(drop=True)
    return df


def load_data(args, share_memory=False):
    ''' load the training and validation set, and initialize the prediction problem'''
    XVARS = args.xvars
    MODEL = args.model
    if args.finalmodel:
        ts = loadcsv(MODEL, TRAIN)
        ts = pd.concat([ts, loadcsv(MODEL, VALID)],
                       ignore_index=True).reset_index(drop=True)
    else:
        ts = loadcsv(MODEL, TRAIN)
    gc.collect()
    print(ts.shape)
    varxfloat = choosevar(vargroupfloat, XVARS)
    varxcat = choosevar(vargroupcat, XVARS)

    for v in varxcat:
        print(v, np.sum(ts[v] == 0)/ts.shape[0])
    featuretarget = predictionproblem.PredictionProblem(args,
        varxfloat, varxcat, yvars(args), ts)
    train_set = torch.utils.data.TensorDataset(*featuretarget.build(ts))
    del ts
    gc.collect()
    if args.finalmodel:
        vs = loadcsv(MODEL, TEST)
    else:
        vs = loadcsv(MODEL, VALID)
    valid_set = torch.utils.data.TensorDataset(*featuretarget.build(vs))
    del vs

    print(gc.collect())
    if share_memory:
        for s in [train_set,  valid_set]:
            for x in s.tensors:
                x.share_memory_()
    return featuretarget, train_set, valid_set


def performtraining(args, device, data, best=None):
    ''' perform training steps'''
    featuretarget, train_set, valid_set = data
    train_loader, trainvalid_loader, valid_loader = data_loader(
        args, train_set, valid_set)
#    torchmseLoss = torch.nn.MSELoss(reduction="elementwise_mean").to(device)
#    cmodel model, dicoembedd, formatx, modelfinal
    cmodel = predictionproblem.create_model(
        device, args, featuretarget, trainvalid_loader, len(yvars(args)))
    criterion = gen_nll(cmodel.outputLayer.normalized_nll,args.ynoise)
    criterion_nonoise = gen_nll(cmodel.outputLayer.normalized_nll)
    def rmse(x):
#        print(featuretarget.std)
        return torch.sqrt(x)*featuretarget.std
    criterionprint = gen_mtlLoss(cmodel.outputLayer.mean)
    # def criterionprint(*x):
    #     loss=criterionprintbeforerescale(*x)
    #     print("criterionprintbeforerescale(*x)",criterionprintbeforerescale(*x))
    #     return rmse(criterionprintbeforerescale(*x)).cpu()# [rmse(loss).cpu() for loss in criterionprintbeforerescale(*x)]
    optimizer = predictionproblem.create_optimizer(args, cmodel)
#    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.rate)
    nbatch_in_train = len(train_loader)//args.batch_size
    args.freqvalid = min(nbatch_in_train, MAX_FREQVALID) if args.freqvalid is None else args.freqvalid
    del nbatch_in_train
    step_size_up = args.freqvalid//2
    step_size_down = args.freqvalid - step_size_up
#    freqvalid = min(nbatch_in_train, args.freqvalid)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,args.initrate*args.factorlowertriangular,args.initrate,cycle_momentum=False,mode=args.modecyclic,scale_mode='cycle',step_size_up=step_size_up,step_size_down=step_size_down)
    earlystop = EarlyStop(20)
    dargs = sorted(vars(args).items())
    logging = CsvLog(args.filelog)
    logging.add2line(map(lambda x: x[0], dargs))
    sets = [valid_loader] if args.finalmodel else [trainvalid_loader, valid_loader]
    colnameset = ["test"] if args.finalmodel else ["train","valid"]
    logging.add2line(["countbackprop", "time", "stop"] + colnameset)
    logging.writeline()


    countbackprop = 0
    countvalid = 0
    train_loader_cycle = Infinitecycle(train_loader)


    start = time.time()
    while countbackprop < args.niteration:
        startepoch = time.perf_counter()
        nmax = min(args.freqvalid, args.niteration-countbackprop)
        predictionproblem.train(device, featuretarget, train_loader_cycle, cmodel.model,
                                criterion, optimizer, scheduler, niter=nmax, clip=args.clip)
        countbackprop += nmax
        endepoch = time.perf_counter()
        print("backward passes",endepoch-startepoch)
        startepoch = time.perf_counter()
        with torch.no_grad():
            losses = [predictionproblem.valid_loss(
                device, featuretarget, s, cmodel.model, (criterion_nonoise,criterionprint)) for s in sets]
        endepoch = time.perf_counter()
        print("compute statistics",endepoch-startepoch)
        lossesprint = [ losses[i][1] for i,_ in enumerate(sets)]
        losses = [ losses[i][0] for i,_ in enumerate(sets)]
        countvalid += 1
        convertedloss = [rmse(loss).cpu() for loss in lossesprint]
        meanlosses = [loss.item() for loss in losses]
#        losses = [[x.item() for x in loss.cpu()] for loss in losses]
        preambule = "iter {:d} seed {:d}".format(countbackprop, args.seed)
        print(preambule, convertedloss, meanlosses, sep="\n")

        def writeline(stop=""):
            logging.add2line(map(lambda x: x[1], dargs))
            logging.add2line([countbackprop, time.time()-start,
                              stop] + meanlosses)
            logging.writeline()

        if not args.finalmodel:
            score = meanlosses[-1]
            if earlystop.step(score):
                writeline("earlystop")
                break
            if best is not None:
                if best[countvalid-1] + 2*args.tolbest < score:
                    writeline("bad")
                    break
                if countvalid-1 >= 4 and best[countvalid-1] + args.tolbest < score:
                    writeline("bad")
                    break
                best[countvalid-1] = min(best[countvalid-1], score)
            if scheduler.get_lr()[0] < args.thresholdlrstop:
                writeline("learningratetoolowtocontinue")
                break
        writeline()
    logging.close()
    return cmodel#.dico, cmodel.model


def data_loader(args, train_set, valid_set):
    ''' Builds the data_loader from the sets'''
    pin_memory = True
    num_workers = 0
    train_loader = tensorDataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    trainvalid_loader = tensorDataLoader(train_set, batch_size=BATCH_SIZE_VALID,
                                         shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = tensorDataLoader(valid_set, batch_size=BATCH_SIZE_VALID,
                                    shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, trainvalid_loader, valid_loader


def train(device, data, args, best=None):
    '''train the network and save it if necessary'''
    start = time.time()
    print("args.seed", args.seed)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    featuretarget, train_set, valid_set = data
    train_loader, trainvalid_loader, valid_loader = data_loader(
        args, train_set, valid_set)
    featuretarget.std = featuretarget.std.to(device)
    if args.initrate is None:
        args.initrate = predictionproblem.search_initrate(
            args, device, featuretarget, train_loader, trainvalid_loader, len(yvars(args)))
    print("initial learning rate", args.initrate)
    cmodel = performtraining(args, device, data, best)
    if args.save_model is not None:
        print("saving model:", args.save_model)
        with open(args.save_model, "wb") as f:
            pickle.dump(TrainResult(
                args.model, featuretarget, cmodel), f)
    end = time.time()
    print(end-start)


def main():
    parser = fargs()
    args = parser.parse_args()
    device = torch.device(
        "cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    print("device used ", device)
    data = load_data(args)
    train(device, data, args)


if __name__ == '__main__':
    main()
