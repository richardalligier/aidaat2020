import argparse
import numpy as np


def testrandomgen(args,n):
    '''generate all the random hyperparameters to test'''

    # number of layers
    nlayers = np.random.randint(1, 11, n)

    def generatearchirandom():
        a = []
        possiblehidden = list(range(10, 100, 10)) + list(range(100, 800, 100))
        for i in range(n):
            w = [np.random.randint(0, len(possiblehidden))
                 for _ in range(nlayers[i])]
            archi = "_".join([str(possiblehidden[x])
                              for x in sorted(w, reverse=True)])
            a.append(archi)
        return a
    r = {}
    optional = {}
    batches = [512, 1024]
    # architecture of the network (number of layers and number of hidden units)
    r["-archi"] = generatearchirandom()
    r["-target"] = np.repeat("all", n)
    r["-niteration"] = np.repeat(5*10**5, n)
    r["-seed"] = np.repeat(0, n)
    # no dropout (set to 0)
    r['-dropout'] = np.repeat(0, n)
    r["-batch_size"] = [batches[np.random.randint(len(batches))] for _ in range(n)]
    # use of dropout2d
    r['-dropout2d'] = np.repeat("", n)
    # dropout rate for the embeddings
    r["-dropout_emb"] = np.random.uniform(0., 0.9, n)
    # size of the embeddings
    r["-sum_emb"] = [np.random.randint(1, 10) for _ in range(n)]
    # weight decay
    r["-lamb"] = 10 ** np.random.uniform(-3., 0., n)
    # number of gaussian inside the mixture
    r["-nmixture"] = [5*np.random.randint(1, 5) for _ in range(n)]
    # dont forget to add clip in selecthyper
    r["-clip"] = np.repeat(0.2, n)
    optional["-pca"] = np.repeat(1 if args.pca else 0,n)
    optional["-whiten"] =  np.repeat(1 if args.whiten else 0,n)
    return r, optional


def printres(ro):
    r, o = ro
    res = []
    n = len(r[list(r)[0]])
    for i in range(n):
        cmd = " ".join([k + " " + str(r[k][i]) for k in r] + [ k for k in o if o[k][i]==1])
        res.append(cmd)
    print("\n".join(res))


def fargs():
    parser = argparse.ArgumentParser(description='generate hyperparameters to test')
    parser.add_argument('-seed',default=0,type=int)
    parser.add_argument('-pca', help='pca', action='store_true', default=False)
    parser.add_argument('-whiten', help='whiten', action='store_true', default=False)
    return parser

if __name__ == '__main__':
    parser = fargs()
    args = parser.parse_args()
    np.random.seed(args.seed)
    # 2000 random hyperparameters but actually, the '-iend 200' option makes 'batchtrain.py' only use the first 200 hyperparameters
    printres(testrandomgen(args,2000))
