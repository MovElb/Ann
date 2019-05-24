import logging
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from bertynet.BertyNet import BertyNet
from bertynet.utils import AverageMeter

logger = logging.getLogger(__name__)


class BertyModel:
    def __init__(self, opt, embeddings=None, state_dict=None):
        self.opt = opt
        self.embeddings = embeddings
        self.averaged_loss = AverageMeter()
        self.iterations = state_dict['iterations'] if state_dict is not None else 0

        if state_dict is not None:
            self.averaged_loss.load(state_dict['averaged_loss'])

        self.network = BertyNet(opt, glove_embeddings=embeddings)

        if state_dict is not None:
            new_state_dict = self.network.state_dict()
            new_state_keys = set(new_state_dict.keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state_keys:
                    del state_dict['network'][k]
            for k in new_state_keys:
                if k not in state_dict['network']:
                    state_dict['network'][k] = new_state_dict[k]
            self.network.load_state_dict(state_dict['network'])

        self.opt_state_dict = state_dict['optimizer'] if state_dict is not None else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters, weight_decay=self.opt['weight_decay'])
        if self.opt_state_dict is not None:
            self.optimizer.load_state_dict(self.opt_state_dict)

    def update(self, batch):
        self.network.train()

        prepared_input = self.network.prepare_input(batch)
        logits = self.network(prepared_input)
        loss = self.network.compute_loss(*logits,
                                         prepared_input['answer_start'],
                                         prepared_input['answer_end'],
                                         prepared_input['plaus_answer_start'],
                                         prepared_input['plaus_answer_end'],
                                         prepared_input['has_answer'])
        self.averaged_loss.update(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.opt['grad_clipping'])
        self.optimizer.step()

        self.iterations += 1
        self.network.reset_fixed_embeddings()

    def predict(self, batch):
        self.network.eval()

        prepared_input = self.network.prepare_input(batch, evaluation=True)
        with torch.no_grad():
            logits_s, logits_e, _, _, logits_answerable = self.network(prepared_input)

        # predicting whether the question is answerable
        scores_answerable = F.softmax(logits_answerable, dim=1)
        thresh_fix = self.opt['threshold_ans'] - 0.5
        scores_answerable[:, 0] += thresh_fix
        scores_answerable[:, 1] -= thresh_fix
        is_answerable = torch.argmax(scores_answerable, dim=1)
        is_answerable = is_answerable.tolist()

        scores_s = F.softmax(logits_s, dim=1)[:, 1:]
        scores_e = F.softmax(logits_e, dim=1)[:, 1:]

        max_pred_len = self.opt['max_pred_len'] or scores_s.size(1)
        scores_mat = torch.bmm(scores_s.unsqueeze(2), scores_e.unsqueeze(1))
        scores_mat.triu_().tril_(max_pred_len - 1)
        start_idxs = torch.argmax(torch.max(scores_mat, 2)[0], 1)
        end_idxs = torch.argmax(torch.max(scores_mat, 1)[0], 1)

        contexts = batch[11]
        spans = batch[10]
        predictions = []
        for i in range(start_idxs.size(0)):
            start_idx, end_idx = start_idxs[i].item(), end_idxs[i].item()
            start_offset, end_offset = spans[i][start_idx][0], spans[i][end_idx][1]
            predictions.append(contexts[i][start_offset:end_offset])

        return predictions, is_answerable

    def infer(self, batch):
        self.network.eval()
        prepared_input = self.network.prepare_input(batch, evaluation=True)
        with torch.no_grad():
            logits_s, logits_e, logits_ps, logits_pe, logits_answerable = self.network(prepared_input)
        scores_answerable = F.softmax(logits_answerable, dim=1)
        thresh_fix = self.opt['threshold_ans'] - 0.5
        scores_answerable[:, 0] += thresh_fix
        scores_answerable[:, 1] -= thresh_fix
        is_answerable = torch.argmax(scores_answerable, dim=1)
        has_answer_score = scores_answerable[:, 1].tolist()

        scores_s = F.softmax(logits_s, dim=1)[:, 1:]
        scores_e = F.softmax(logits_e, dim=1)[:, 1:]

        scores_ps = F.softmax(logits_ps, dim=1)[:, 1:]
        scores_pe = F.softmax(logits_pe, dim=1)[:, 1:]

        max_pred_len = self.opt['max_pred_len'] or scores_s.size(1)

        scores_mat = torch.bmm(scores_s.unsqueeze(2), scores_e.unsqueeze(1))
        scores_mat.triu_().tril_(max_pred_len - 1)
        start_scores, start_idxs = torch.max(torch.max(scores_mat, 2)[0], 1)
        end_scores, end_idxs = torch.max(torch.max(scores_mat, 1)[0], 1)

        scores_pmat = torch.bmm(scores_ps.unsqueeze(2), scores_pe.unsqueeze(1))
        scores_pmat.triu_().tril_(max_pred_len - 1)
        start_pscores, start_pidxs = torch.max(torch.max(scores_pmat, 2)[0], 1)
        end_pscores, end_pidxs = torch.max(torch.max(scores_pmat, 1)[0], 1)

        contexts = batch[11]
        spans = batch[10]
        predictions = []
        plausible_predictions = []
        for i in range(start_idxs.size(0)):
            start_idx, end_idx = start_idxs[i].item(), end_idxs[i].item()
            start_offset, end_offset = spans[i][start_idx][0], spans[i][end_idx][1]

            start_pidx, end_pidx = start_pidxs[i].item(), end_pidxs[i].item()
            start_poffset, end_poffset = spans[i][start_pidx][0], spans[i][end_pidx][1]

            predictions.append(contexts[i][start_offset:end_offset])
            plausible_predictions.append(contexts[i][start_poffset:end_poffset])

        score = start_scores * end_scores
        plausible_score = start_pscores * end_pscores

        return {
            'predictions': predictions,
            'plausible_predictions': plausible_predictions,
            'score': score.tolist(),
            'plausible_score': plausible_score.tolist(),
            'is_answerable': is_answerable.tolist(),
            'has_ans_score': has_answer_score
        }

    def save(self, filename, epoch):
        state_dict = self.network.state_dict()
        updated_state_dict = state_dict.copy()
        for key in state_dict.keys():
            if 'bert' in key:
                del updated_state_dict[key]
        params = {
            'state_dict': {
                'network': updated_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'iterations': self.iterations,
                'averaged_loss': self.averaged_loss.state_dict()
            },
            'config': self.opt,
            'epoch': epoch,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        try:
            torch.save(params, filename)
            logger.info('Model has been saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
        # we need to rebuild the optimizer according to
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336183179
        self.build_optimizer()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
