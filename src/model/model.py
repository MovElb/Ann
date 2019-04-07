import torch
import torch.optim as optim
from .utils import AverageMeter
from .BertyNet import BertyNet


class BertyModel():
    def __init__(self, opt, embeddings=None, state_dict=None):
        self.opt = opt
        self.embeddings = embeddings
        self.averaged_loss = AverageMeter()
        self.iterations = state_dict['iterations'] if state_dict is not None else 0
        if state_dict is not None:
            self.averaged_loss.load(state_dict['averaged_loss'])

        self.network = BertyNet(opt, glove_embeddings=embeddings)
        if state_dict is not None:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        self.opt_state_dict = state_dict['optimizer'] if state_dict is not None else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters, weight_decay=self.opt['weight_decay'])
        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)
