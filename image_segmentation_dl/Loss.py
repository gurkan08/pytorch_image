from torch.nn import BCELoss

class Loss(object):

    def get_loss(loss_name):
        if loss_name=="bce":
            return BCELoss()