import torch


def batch_loss(device, featuretarget, xy, output, loss):
    '''utility function to compute the loss of a batch'''
    target = featuretarget.target(xy).to(device)
    weight = featuretarget.weights(xy).to(device)
#    print(weight.shape)
    n = weight.sum(0)
    err = loss(output, target) * weight
    loss = err.sum(0)
    return loss, n



def gen_nll(get_nll, noise=None):
    def batch_loss(device, featuretarget, xy, output):
        '''utility function to compute the loss of a batch'''
        target = featuretarget.target(xy).to(device)
#        print(target.std(0))
        if noise is not None:
            target = target + noise * torch.randn_like(target)
        # else:
        #     print("no noise")
            #        weight = featuretarget.weights(xy).to(device)
        n = torch.tensor([target.shape[0]],dtype=torch.float)#weight.sum(0)
        err = get_nll(output, target) #* weight
        loss = err.sum(0)
#        print("loss.shape,n.shape",loss.shape,n.shape)
        return loss, n
    return batch_loss


_mseloss = torch.nn.MSELoss(reduction='none')
def gen_mtlLoss(get_mean):
    def mtlLoss(device, featuretarget, xy, output):
        ''' MSE loss on the first output (the output is a tuple of 2 tensor)'''
        def mseloss(output, target):
            meanoutput = get_mean(output)
            return _mseloss(meanoutput, target)
        return batch_loss(device, featuretarget, xy, output, mseloss)
    return mtlLoss


# def pnllNormal(device, featuretarget, xy, output):
#     '''predictive negative log-likelihood '''
#     def pnll(output, target):
#         meanoutput, sig2output = output
#         return (meanoutput-target)**2/sig2output+torch.log(sig2output)
#     return batch_loss(device, featuretarget, xy, output, pnll)
