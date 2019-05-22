import re
import unicodedata
import collections


def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text).strip()
    return text


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


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


def annotate(row, wv_cased, init_nlp=None):
    context, question = row[:2]
    question_features = annotate_single(question, init_nlp=init_nlp)
    context_features = annotate_single(context, init_nlp=init_nlp)
    return annotate_crossed(context_features, question_features, wv_cased) + row[2:]


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
