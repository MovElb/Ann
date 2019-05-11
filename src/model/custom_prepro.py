import msgpack
import numpy as np
import spacy
from prepro import annotate, to_id, normalize_text
from functools import partial
import logging

class CustomPrepro():
    def __init__(self
                , meta_file='./meta.msgpack'
                , wv_file='../../data/glove.840B.300d.txt'
                , wv_dim=300):
        with open(meta_file, 'rb') as f:
            meta = msgpack.unpack(f)
        self.vocab = meta['vocab']
        self.vocab_tag = meta['vocab_tag']
        self.vocab_ent = meta['vocab_ent']
        self.embeddings = meta['embedding']
        self.wv_cased = meta['wv_cased']

        self.w2id = {w: i for i, w in enumerate(self.vocab)}
        self.tag2id = {w: i for i, w in enumerate(self.vocab_tag)}
        self.ent2id = {w: i for i, w in enumerate(self.vocab_ent)}

        self.wv_vocab = set()
        self.wv_file = wv_file
        self.wv_dim = wv_dim

        with open(self.wv_file) as f:
            for line in f:
                token = normalize_text(line.rstrip().split(' ')[0])
                self.wv_vocab.add(token)

        self.nlp = spacy.load('en', parser=False)
        self.annotate_ = partial(annotate, wv_cased=self.wv_cased, init_nlp=self.nlp)
        self.to_id_ = partial(to_id, w2id=self.w2id, tag2id=self.tag2id, ent2id=self.ent2id)

    def prepro(self, context, question):
        raw = (context, question)
        raw = self.annotate_(raw)
        raw = self.to_id_(raw)
        embeddings_context = np.array([self.embeddings[token_id] for token_id in raw[0]])
        embeddings_question = np.array([self.embeddings[token_id] for token_id in raw[5]])
        return raw, embeddings_context, embeddings_question
