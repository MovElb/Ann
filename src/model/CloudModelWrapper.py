import os
import random
from functools import partial

import torch

import msgpack
from model import BertyModel
from BatchGen import BatchGen


class CloudModelWrapper:
    def __init__(self, checkpoint_filepath, args):
        checkpoint = torch.load(checkpoint_filepath, map_location="cuda" if args.use_cuda else "cpu")
        opt = checkpoint['config']
        opt.update(vars(args))

        self.state_dict = checkpoint['state_dict']
        self.opt = opt
        self.embeddings = None
        self.load_embeddings_and_update_opt_()

        self.model = BertyModel(self.opt, self.embeddings, self.state_dict)
        # synchronize random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'].cpu())
        if args.use_cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'].cpu())

        self.bg_wrap = partial(BatchGen, batch_size=1, evaluation=True)

    def load_embeddings_and_update_opt_(self):
        meta_filename = os.path.join(self.opt['data_dir'], self.opt['meta_file'])
        with open(meta_filename, 'rb') as f:
            meta = msgpack.load(f, encoding='utf8')

        self.embeddings = meta['embedding']
        self.opt['pos_size'] = len(meta['vocab_tag'])
        self.opt['ner_size'] = len(meta['vocab_ent'])

    def generate_model_answers(self, preprocessed_data):
        batched_data = next(iter(self.bg_wrap([preprocessed_data])))
        model_answers = self.model(batched_data)
        return model_answers


