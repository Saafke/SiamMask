import torch.nn as nn


class Mask(nn.Module):

    def __init__(self):
        """I am a function."""
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        """I am a function."""
        raise NotImplementedError

    def template(self, template):
        """I am a function."""
        raise NotImplementedError

    def track(self, search):
        """I am a function."""
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        """I am a function."""
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params
