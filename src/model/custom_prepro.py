import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from prepro import annotate, annotate_single, annotate_crossed, to_id
from multiprocessing import Pool
from utils import str2bool
from tqdm import tqdm
from functools import partial
import logging

class CustomPrepro():
    def __init__(self
                , meta_file='./squad2_preprocessed/meta.msgpack'
                , wv_dim=300):
        with open(meta_file, 'rb') as f:
            meta = msgpack.unpack(f, encoding='utf8')
        self.vocab = meta['vocab']
        self.vocab_tag = meta['vocab_tag']
        self.vocab_ent = meta['vocab_ent']
        self.embeddings = meta['embedding']
        self.wv_cased = meta['wv_cased']

        self.w2id = {w: i for i, w in enumerate(self.vocab)}
        self.tag2id = {w: i for i, w in enumerate(self.vocab_tag)}
        self.ent2id = {w: i for i, w in enumerate(self.vocab_ent)}

        self.nlp = spacy.load('en', parser=False)
        self.annotate = partial(annotate, wv_cased=self.wv_cased, init_nlp=self.nlp)
        self.annotate_single = partial(annotate_single, init_nlp=self.nlp)
        self.annotate_crossed = partial(annotate_crossed, wv_cased=self.wv_cased)
        self.to_id = partial(to_id, w2id=self.w2id, tag2id=self.tag2id, ent2id=self.ent2id)

    def prepro(self, context, question):
        raw_data = (context, question)
        preprocessed_data = self.annotate(raw_data)
        preprocessed_data = self.to_id(preprocessed_data)
        return preprocessed_data

    def prepro_text(self, text):
        return self.annotate_single(text)

    def prepro_crossed(self, context_features, question_features):
        preprocessed_data = self.annotate_crossed(context_features, question_features)
        preprocessed_data = self.to_id(preprocessed_data)
        return preprocessed_data
