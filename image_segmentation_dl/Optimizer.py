from torch import optim

class Optimizer(object):

    def get_optimizer(opt_name, model, lr):
        if opt_name=="sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        if opt_name=="adam":
            return optim.Adam(model.parameters(), lr=lr)