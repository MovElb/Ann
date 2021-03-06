import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from multiprocessing import Pool
from bertynet.utils import str2bool
from tqdm import tqdm
from functools import partial
import logging

def main():
    args, log = setup()

    train = flatten_json(args.trn_file, 'train')
    dev = flatten_json(args.dev_file, 'dev')
    log.info('json data flattened.')

    # tokenize & annotate
    with Pool(args.threads, initializer=init) as p:
        annotate_ = partial(annotate, wv_cased=args.wv_cased)
        train = list(tqdm(p.imap(annotate_, train, chunksize=args.batch_size), total=len(train), desc='train'))
        dev = list(tqdm(p.imap(annotate_, dev, chunksize=args.batch_size), total=len(dev), desc='dev  '))
    train = list(map(index_answer, train))
    initial_len = len(train)
    train = list(filter(lambda x: x[-1] is not None, train))
    log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
    log.info('tokens generated')

    # load vocabulary from word vector files
    wv_vocab = set()
    with open(args.wv_file) as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            wv_vocab.add(token)
    log.info('glove vocab loaded.')

    # build vocabulary
    full = train + dev
    vocab, counter = build_vocab([row[4] for row in full], [row[0] for row in full], wv_vocab, args.sort_all)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    counter_tag_c = collections.Counter(w for row in full for w in row[2])
    counter_tag_q = collections.Counter(w for row in full for w in row[6])
    counter_tag = counter_tag_c + counter_tag_q
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent_c = collections.Counter(w for row in full for w in row[3])
    counter_ent_q = collections.Counter(w for row in full for w in row[7])
    counter_ent = counter_ent_c + counter_ent_q
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    w2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    train = list(map(to_id_, train))
    dev = list(map(to_id_, dev))
    log.info('converted to ids.')

    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, args.wv_dim))
    embed_counts = np.zeros(vocab_size)
    embed_counts[:2] = 1  # PADDING & UNK
    with open(args.wv_file) as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = normalize_text(elems[0])
            if token in w2id:
                word_id = w2id[token]
                embed_counts[word_id] += 1
                embeddings[word_id] += [float(v) for v in elems[1:]]
    embeddings /= embed_counts.reshape((-1, 1))
    log.info('got embedding matrix.')

    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist(),
        'wv_cased': args.wv_cased,
    }
    with open('./squad2_preprocessed/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)
    result = {
        'train': train,
        'dev': dev
    }

    # train:
    #     0: context_ids, 1: context_tokens, 2: context_features,
    #     3: context_tag_ids, 4: context_ent_ids,
    #     5: question_ids, 6: question_tokens, 7: question_features,
    #     8: question_tag_ids, 9: question_ent_ids,
    #     10: context_token_span, 11: context, 12: question, 13: has_ans,
    #     14: answer_start, 15: answer_end,
    #     16: plausible_answer_start, 17: plausible_answer_end
    # dev:
    #     0: context_ids, 1: context_tokens, 2: context_features,
    #     3: context_tag_ids, 4: context_ent_ids,
    #     5: question_ids, 6: question_tokens, 7: question_features,
    #     8: question_tag_ids, 9: question_ent_ids,
    #     10: context_token_span, 11: context, 12: question,
    #     13: has_ans, 14: answer

    with open('./squad2_preprocessed/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)
    if args.sample_size:
        sample = {
            'train': train[:args.sample_size],
            'dev': dev[:args.sample_size]
        }
        with open('./squad2_preprocessed/sample.msgpack', 'wb') as f:
            msgpack.dump(sample, f)
    log.info('saved to disk.')

def setup():
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    parser.add_argument('--trn_file', default='./squad2_data/train-v2.0.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='./squad2_data/dev-v2.0.json',
                        help='path to dev file.')
    parser.add_argument('--wv_file', default='./squad2_data/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='treat the words as cased or not.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words. '
                             'Otherwise consider question words first.')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='size of sample data (for debugging).')
    parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 16),
                        help='number of threads for preprocessing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for multiprocess tokenizing and tagging.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')

    return args, log

def flatten_json(data_file, mode):
    """Flatten each article in training data."""
    with open(data_file) as f:
        data = json.load(f)['data']
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                has_ans = not qa['is_impossible']
                answers = qa['answers'] if has_ans else qa['plausible_answers']
                id_, question = qa['id'], qa['question']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start'] if has_ans else 0
                    answer_end = answer_start + len(answer) if has_ans else 0
                    p_answer_start = answers[0]['answer_start']
                    p_answer_end = p_answer_start + len(answer)
                    rows.append((context, question, has_ans, answer, answer_start, answer_end, p_answer_start, p_answer_end))
                else:  # mode == 'dev'
                    answers = [a['text'] for a in answers]
                    rows.append((context, question, has_ans, answers))
    return rows

def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text).strip()
    return text

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

nlp = None

def init():
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)

def annotate_single(text, init_nlp=None):
    if init_nlp is None:
        global nlp
    else:
        nlp = init_nlp
    doc = nlp(clean_spaces(text))
    tokens = [normalize_text(w.text) for w in doc]
    tokens_lower = [w.lower() for w in tokens]
    token_span = [(w.idx, w.idx + len(w.text)) for w in doc]
    tags = [w.tag_ for w in doc]
    ents = [w.ent_type_ for w in doc]
    # term frequency in document, context
    counter = collections.Counter(tokens_lower)
    total = len(tokens_lower)
    tf = [counter[w] / total for w in tokens_lower]
    return (text, doc, tokens, tokens_lower, tags, ents, tf, token_span)

def annotate_crossed(context_features, question_features, wv_cased):
    question, q_doc, question_tokens, question_tokens_lower, \
        question_tags, question_ents, question_tf, _ = question_features
    context, c_doc, context_tokens, context_tokens_lower, \
        context_tags, context_ents, context_tf, context_token_span = context_features

    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    context_match_origin = [w in question_tokens_set for w in context_tokens]
    context_match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    context_match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # features: origin, lower, lemma for question-context
    context_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in c_doc}
    context_tokens_set = set(context_tokens)
    context_tokens_lower_set = set(context_tokens_lower)
    question_match_origin = [w in context_tokens_set for w in question_tokens]
    question_match_lower = [w in context_tokens_lower_set for w in question_tokens_lower]
    question_match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in context_lemma for w in q_doc]

    context_features = list(zip(context_match_origin, context_match_lower, context_match_lemma, context_tf))
    question_features = list(zip(question_match_origin, question_match_lower, question_match_lemma, question_tf))
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (context_tokens, context_features, context_tags, context_ents,
            question_tokens, question_features, question_tags, question_ents,
            context_token_span, context, question)

def annotate(row, wv_cased, init_nlp=None, calc_cross_features=False):
    context, question = row[:2]
    question_features = annotate_single(question, init_nlp=init_nlp)
    context_features = annotate_single(context, init_nlp=init_nlp)
    return annotate_crossed(context_features, question_features, wv_cased) + row[2:]

def index_answer(row):
    token_span = row[-9]
    starts, ends = zip(*token_span)
    answer_start, answer_end = row[-4], row[-3]
    p_answer_start, p_answer_end = row[-2], row[-1]
    try:
        return row[:-5] + (starts.index(answer_start), ends.index(answer_end), starts.index(p_answer_start), ends.index(p_answer_end))
    except ValueError:
        pass
    try:
        return row[:-5] + (0, 0, starts.index(p_answer_start), ends.index(p_answer_end))
    except ValueError:
        return row[:-5] + (None, None, None, None)

def build_vocab(questions, contexts, wv_vocab, sort_all=False):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter

def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens, context_features, context_tags, context_ents, \
        question_tokens, question_features, question_tags, question_ents = row[:8]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_tag_ids = [tag2id[w] for w in context_tags]
    question_tag_ids = [tag2id[w] for w in question_tags]
    context_ent_ids = [ent2id[w] for w in context_ents]
    question_ent_ids = [ent2id[w] for w in question_ents]
    return (context_ids, context_tokens, context_features, context_tag_ids, context_ent_ids,
        question_ids, question_tokens, question_features, question_tag_ids, question_ent_ids) + row[8:]

if __name__ == '__main__':
    main()
