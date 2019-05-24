import argparse
import re
import string
from collections import Counter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter:
    """Keep exponential weighted averages."""

    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, has_ans, answers_true, has_ans_true):
    if pred is None or answers_true is None:
        return False
    if not has_ans or not has_ans_true:
        return has_ans == has_ans_true
    pred = _normalize_answer(pred)
    for a in answers_true:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, has_ans, answers_true, has_ans_true):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers_true is None:
        return 0
    if not has_ans or not has_ans_true:
        return has_ans == has_ans_true
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers_true]
    return max(scores)


def score(predictions, is_answerable, dev_answer_info):
    is_answerable_truth, answers_truth = list(zip(*dev_answer_info))
    assert len(predictions) == len(answers_truth)
    f1 = em = total = 0
    for pred, has_ans, ans_true, has_ans_true, in zip(predictions, is_answerable, answers_truth, is_answerable_truth):
        total += 1
        em += _exact_match(pred, has_ans, ans_true, has_ans_true)
        f1 += _f1_score(pred, has_ans, ans_true, has_ans_true)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1
