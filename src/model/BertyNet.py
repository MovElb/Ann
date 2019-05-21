import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer

from layers import FullAttention, StackedBRNN, Summarize, PointerNet


class BertyNet(nn.Module):
    def __init__(self, opt, glove_embeddings=None):
        """
        Args:
            glove_embeddings (list of lists): Matrix of Glove embeddings with shape (vocab_len, embedding_dim) or None
        """
        super(BertyNet, self).__init__()
        self.use_cuda = opt['use_cuda']
        self.opt = opt
        self.g_pretr_embeddings = glove_embeddings
        self._build_model()

    def _build_model(self):
        GLOVE_DIM = self.opt['glove_dim']
        BERT_DIM = self.opt['bert_dim']
        POS_DIM = self.opt['pos_dim']
        POS_SIZE = self.opt['pos_size']
        NER_DIM = self.opt['ner_dim']
        NER_SIZE = self.opt['ner_size']
        CUSTOM_FEAT_DIM = 4
        TOTAL_DIM = GLOVE_DIM + BERT_DIM + POS_DIM + NER_DIM + CUSTOM_FEAT_DIM

        RNN_HIDDEN_SIZE = self.opt['rnn_hidden_size']
        ATTENTION_HIDDEN_SIZE = self.opt['attention_hidden_size']
        DROPOUT_RATE = self.opt['dropout_rate']

        if self.g_pretr_embeddings is not None:
            glove_pretr_embeddings = torch.tensor(self.g_pretr_embeddings)
            self._glove_embeddings = nn.Embedding(
                    glove_pretr_embeddings.size(0),
                    glove_pretr_embeddings.size(1),
                    padding_idx=0)
            self._glove_embeddings.weight.data[2:, :] = glove_pretr_embeddings[2:, :]
            if self.opt['tune_partial'] > 0:
                assert self.opt['tune_partial'] + 2 < glove_pretr_embeddings.size(0)
                glove_fixed_embeddings = glove_pretr_embeddings[self.opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embeddings', glove_fixed_embeddings)
                self._glove_fixed_embeddings = glove_fixed_embeddings

        else:
            self._glove_embeddings = nn.Embedding(self.opt['vocab_size'], GLOVE_DIM, padding_idx=0)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if self.use_cuda:
            self.bert_model.cuda()
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.max_len = 2048

        self._pos_embeddings = nn.Embedding(POS_SIZE, POS_DIM, padding_idx=0)
        self._ner_embeddings = nn.Embedding(NER_SIZE, NER_DIM, padding_idx=0)

        self._word_attention = FullAttention(GLOVE_DIM, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE,
                                             use_cuda=self.use_cuda)

        init_input_size = TOTAL_DIM + GLOVE_DIM
        self._universal_node = nn.Parameter(torch.zeros(1, init_input_size))
        nn.init.xavier_normal_(self._universal_node)

        cur_input_size = init_input_size
        self._low_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._high_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * (2 * RNN_HIDDEN_SIZE)
        self._full_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        attention_input_size = 2 * (2 * RNN_HIDDEN_SIZE) + init_input_size
        self._low_attention = FullAttention(
                attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._high_attention = FullAttention(
                attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._full_attention = FullAttention(
                attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)

        cur_input_size = 6 * (2 * RNN_HIDDEN_SIZE)
        self._fusion_lstm = StackedBRNN(
                cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        self_attention_input_size = 2 * RNN_HIDDEN_SIZE + init_input_size
        self._self_attention = FullAttention(
                self_attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)

        cur_input_size = self_attention_input_size + 2 * RNN_HIDDEN_SIZE
        self._final_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        self._final_plausible_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._question_summarization = Summarize(cur_input_size, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._question_plausible_summarization = Summarize(
                cur_input_size, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._question_verifier_summarization = Summarize(
                cur_input_size, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)

        self._answer_pointer = PointerNet(cur_input_size)
        self._plausible_answer_pointer = PointerNet(cur_input_size)

        cur_input_size = 4 * (2 * RNN_HIDDEN_SIZE)
        self._answer_verifier = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(cur_input_size, 2))

    def prepare_input(self, batch_data, evaluation=False):
        """Converts token ids to glove, bert, pos, ner embeddings and concatenates all features.
        Injects universal node (trainable parameter) into the batch data between question and context.
        Increases all answer spans which corresponds to answerable examples in order to compute correct loss.

        Args:
            batch_data (list of lists): Batch yielded by BatchGen object. See BatchGen for details.
            evaluation (bool): Flag indicating current mode.

        Returns:
            dict containing:
                ['cat_input'] - Tensor, (B * L * output_dim), B - batch size,
                    L - maximum (probably, padded) length of <question|uni_node|context> sequence,
                    output_dim - dimension of all concatenated features per token.
                ['cat_mask'] - Tensor, (B * L), mask for all RNN layers with all zeros for
                    <question|uni_node> part, and correct mask for <context> part.
                ['question_mask'] - Tensor, (B * (question_len + 1)), mask for <question|uni_node> part.
                ['context_mask - Tensor, (B * (1 + context_len)), mask for <uni_node|context> part.
                ['question_len'] - int, maximum question len for this batch (without uni_node).
                ['answer_start] - Tensor, (B,) or None when evaluation.
                ['answer_end'] - Tensor, (B,) or None when evaluation.
                ['plaus_answer_start'] - Tensor, (B,) or None when evaluation.
                ['plaus_answer_end'] - Tensor, (B,) or None when evaluation.
                ['has_ans'] - Tensor, (B,) or None when evaluation.
        """
        bert_question_embeddings, bert_context_embeddings, new_question_lengths, \
        new_context_lengths, mask_good_ex = self._get_bert_embeddings_for_batch(batch_data, evaluation=evaluation)

        new_question_maxlen = max(new_question_lengths)
        new_context_maxlen = max(new_context_lengths)
        new_batch_size = sum(mask_good_ex)

        assert len(bert_question_embeddings) == new_batch_size

        question_glove_ids = torch.zeros(new_batch_size, new_question_maxlen, dtype=torch.long)
        context_glove_ids = torch.zeros(new_batch_size, new_context_maxlen, dtype=torch.long)

        bert_dim = bert_question_embeddings[0].size(1)
        bert_question_fixed = torch.zeros(new_batch_size, new_question_maxlen, bert_dim)
        bert_context_fixed = torch.zeros(new_batch_size, new_context_maxlen, bert_dim)

        question_pos_ids = torch.zeros(new_batch_size, new_question_maxlen, dtype=torch.long)
        context_pos_ids = torch.zeros(new_batch_size, new_context_maxlen, dtype=torch.long)

        question_ner_ids = torch.zeros(new_batch_size, new_question_maxlen, dtype=torch.long)
        context_ner_ids = torch.zeros(new_batch_size, new_context_maxlen, dtype=torch.long)

        features_len = len(batch_data[2][0][0])
        question_features = torch.zeros(new_batch_size, new_question_maxlen, features_len)
        context_features = torch.zeros(new_batch_size, new_context_maxlen, features_len)

        if self.use_cuda:
            question_glove_ids = question_glove_ids.cuda()
            question_pos_ids = question_pos_ids.cuda()
            question_ner_ids = question_ner_ids.cuda()
            context_glove_ids = context_glove_ids.cuda()
            context_pos_ids = context_pos_ids.cuda()
            context_ner_ids = context_ner_ids.cuda()
            bert_question_fixed = bert_question_fixed.cuda()
            question_features = question_features.cuda()
            bert_context_fixed = bert_context_fixed.cuda()
            context_features = context_features.cuda()

        idx = 0
        for i in range(len(mask_good_ex)):
            if mask_good_ex[i]:
                new_question_ids = torch.tensor(batch_data[5][i], dtype=torch.long)
                new_question_pos_ids = torch.tensor(batch_data[8][i], dtype=torch.long)
                new_question_ner_ids = torch.tensor(batch_data[9][i], dtype=torch.long)
                new_question_features = torch.tensor(batch_data[7][i])
                new_context_ids = torch.tensor(batch_data[0][i], dtype=torch.long)
                new_context_pos_ids = torch.tensor(batch_data[3][i], dtype=torch.long)
                new_context_ner_ids = torch.tensor(batch_data[4][i], dtype=torch.long)
                new_context_features = torch.tensor(batch_data[2][i])

                new_question_ids = new_question_ids[:new_question_lengths[idx]]
                new_question_pos_ids = new_question_pos_ids[:new_question_lengths[idx]]
                new_question_ner_ids = new_question_ner_ids[:new_question_lengths[idx]]
                new_question_features = new_question_features[:new_question_lengths[idx]]
                new_context_ids = new_context_ids[:new_context_lengths[idx]]
                new_context_pos_ids = new_context_pos_ids[:new_context_lengths[idx]]
                new_context_ner_ids = new_context_ner_ids[:new_context_lengths[idx]]
                new_context_features = new_context_features[:new_context_lengths[idx]]

                question_glove_ids[idx, new_question_maxlen - new_question_lengths[idx]:] = new_question_ids
                bert_question_fixed[idx, new_question_maxlen - new_question_lengths[idx]:] = bert_question_embeddings[
                    idx]
                question_pos_ids[idx, new_question_maxlen - new_question_lengths[idx]:] = new_question_pos_ids
                question_ner_ids[idx, new_question_maxlen - new_question_lengths[idx]:] = new_question_ner_ids
                question_features[idx, new_question_maxlen - new_question_lengths[idx]:] = new_question_features
                context_glove_ids[idx, :new_context_lengths[idx]] = new_context_ids
                bert_context_fixed[idx, :new_context_lengths[idx]] = bert_context_embeddings[idx]
                context_pos_ids[idx, :new_context_lengths[idx]] = new_context_pos_ids
                context_ner_ids[idx, :new_context_lengths[idx]] = new_context_ner_ids
                context_features[idx, :new_context_lengths[idx]] = new_context_features

                idx += 1

        node_mask = torch.zeros(new_batch_size, 1, dtype=torch.uint8)
        if self.use_cuda:
            node_mask = node_mask.cuda()

        question_mask = self._compute_mask(question_glove_ids)
        question_mask = torch.cat([question_mask, node_mask], dim=1)

        context_mask = self._compute_mask(context_glove_ids)
        context_mask = torch.cat([node_mask, context_mask], dim=1)

        question_zero_mask = torch.zeros(question_mask.size(0), question_mask.size(1) - 1, dtype=torch.uint8)
        if self.use_cuda:
            question_zero_mask = question_zero_mask.cuda()
        cat_mask = torch.cat(
                [question_zero_mask, context_mask], dim=1)

        question_glove_emb = self._glove_embeddings(question_glove_ids)
        question_pos_emb = self._pos_embeddings(question_pos_ids)
        question_ner_emb = self._ner_embeddings(question_ner_ids)
        context_glove_emb = self._glove_embeddings(context_glove_ids)
        context_pos_emb = self._pos_embeddings(context_pos_ids)
        context_ner_emb = self._ner_embeddings(context_ner_ids)

        cat_question_feat = torch.cat(
                [question_pos_emb, question_ner_emb, question_features], dim=2)
        cat_context_feat = torch.cat(
                [context_pos_emb, context_ner_emb, context_features], dim=2)

        answer_start, answer_end, plaus_answer_start, plaus_answer_end, has_answer = None, None, None, None, None
        if not evaluation:
            selection_mask = torch.tensor(mask_good_ex, dtype=torch.uint8)
            answer_start = torch.tensor(batch_data[14], dtype=torch.long)
            answer_end = torch.tensor(batch_data[15], dtype=torch.long)
            plaus_answer_start = torch.tensor(batch_data[16], dtype=torch.long)
            plaus_answer_end = torch.tensor(batch_data[17], dtype=torch.long)
            has_answer = torch.tensor(batch_data[13], dtype=torch.long)

            answer_start = answer_start.masked_select(selection_mask)
            answer_end = answer_end.masked_select(selection_mask)
            plaus_answer_start = plaus_answer_start.masked_select(selection_mask)
            plaus_answer_end = plaus_answer_end.masked_select(selection_mask)
            has_answer = has_answer.masked_select(selection_mask)

            # fix answer spans in order to consider universal_node at 0 position, so that loss is correctly computed
            has_answer_mask = has_answer.to(torch.uint8)
            has_answer_ind = torch.arange(0, new_batch_size)
            has_answer_ind = has_answer_ind.masked_select(has_answer_mask)

            answer_start[has_answer_ind] += 1
            answer_end[has_answer_ind] += 1
            plaus_answer_start[has_answer_ind] += 1
            plaus_answer_end[has_answer_ind] += 1

            if self.use_cuda:
                answer_start = answer_start.cuda()
                answer_end = answer_end.cuda()
                plaus_answer_start = plaus_answer_start.cuda()
                plaus_answer_end = plaus_answer_end.cuda()
                has_answer = has_answer.cuda()

        return {
            'question_glove': question_glove_emb,
            'context_glove': context_glove_emb,
            'cat_question_feat': cat_question_feat,
            'cat_context_feat': cat_context_feat,
            'question_bert': bert_question_fixed,
            'context_bert': bert_context_fixed,
            'cat_mask': cat_mask,
            'question_mask': question_mask,
            'context_mask': context_mask,
            'question_len': new_question_maxlen,
            'answer_start': answer_start,
            'answer_end': answer_end,
            'plaus_answer_start': plaus_answer_start,
            'plaus_answer_end': plaus_answer_end,
            'has_answer': has_answer
        }

    def _get_bert_embeddings(self, question_tokens, context_tokens, answer_end_idx=None):
        MAX_SEQ_LEN = 512
        CLS_IND = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        SEP_IND = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

        def get_wordpiece_tokenization(tokens):
            UNK = '[UNK]'
            wp_tokens = []
            orig_to_tok_map = []
            for token in tokens:
                wp_tokenized = self.tokenizer.tokenize(token)
                if len(wp_tokenized) == 0:
                    wp_tokenized = [UNK]
                orig_to_tok_map.append(len(wp_tokens))
                wp_tokens.extend(wp_tokenized)
            indexed_wp_tokens = self.tokenizer.convert_tokens_to_ids(wp_tokens)
            if len(indexed_wp_tokens) > MAX_SEQ_LEN - 2:
                indexed_wp_tokens = torch.tensor(indexed_wp_tokens)
                indexed_wp_tokens = indexed_wp_tokens[orig_to_tok_map]
                orig_to_tok_map = list(range(len(indexed_wp_tokens)))
                indexed_wp_tokens = indexed_wp_tokens.tolist()
            return indexed_wp_tokens, orig_to_tok_map

        def truncate_sequence(seq_tokens, mapping, desired_len):
            while len(seq_tokens) > desired_len:
                seq_tokens.pop()
                if mapping[-1] >= len(seq_tokens):
                    mapping.pop()

        def embed_joint(question_wp_tokens, context_wp_tokens, question_mapping, context_mapping):
            cat_tokens = [CLS_IND] + question_wp_tokens + [SEP_IND] + context_wp_tokens + [SEP_IND]
            segment_mask = torch.ones(len(cat_tokens), dtype=torch.long)
            segment_mask[:len(question_wp_tokens) + 2] = 0

            tokens_tensor = torch.tensor([cat_tokens])
            segments_tensor = segment_mask.unsqueeze(0)

            if self.use_cuda:
                tokens_tensor = tokens_tensor.cuda()
                segments_tensor = segments_tensor.cuda()

            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensor)

            sum_last_four_layers = torch.sum(torch.cat(encoded_layers[-4:], dim=0), dim=0)
            question_bert_embeddings = sum_last_four_layers[1: len(question_wp_tokens) + 1]
            question_bert_embeddings = question_bert_embeddings[question_mapping]

            context_bert_embeddings = sum_last_four_layers[len(question_wp_tokens) + 2: -1]
            context_bert_embeddings = context_bert_embeddings[context_mapping]

            return question_bert_embeddings, context_bert_embeddings

        def embed_separately(seq_tokens, seq_mapping):
            cat_tokens = [CLS_IND] + seq_tokens + [SEP_IND]
            segment_mask = torch.zeros(len(cat_tokens), dtype=torch.long)

            tokens_tensor = torch.tensor([cat_tokens])
            segments_tensor = segment_mask.unsqueeze(0)

            if self.use_cuda:
                tokens_tensor = tokens_tensor.cuda()
                segments_tensor = segments_tensor.cuda()

            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensor)

            sum_last_four_layers = torch.sum(torch.cat(encoded_layers[-4:], dim=0), dim=0)
            seq_embeddings = sum_last_four_layers[1: -1]
            seq_embeddings = seq_embeddings[seq_mapping]

            return seq_embeddings

        question_wp_tokens, question_mapping = get_wordpiece_tokenization(question_tokens)
        context_wp_tokens, context_mapping = get_wordpiece_tokenization(context_tokens)

        cat_len = len(question_wp_tokens) + len(context_wp_tokens)
        if cat_len + 3 <= MAX_SEQ_LEN:
            question_embeddings, context_embeddings = embed_joint(question_wp_tokens, context_wp_tokens,
                                                                  question_mapping, context_mapping)
        else:
            truncate_sequence(context_wp_tokens, context_mapping, MAX_SEQ_LEN - 2)
            if answer_end_idx is not None and answer_end_idx >= len(context_mapping):
                return None, None
            truncate_sequence(question_wp_tokens, question_mapping, MAX_SEQ_LEN - 2)

            question_embeddings = embed_separately(question_wp_tokens, question_mapping)
            context_embeddings = embed_separately(context_wp_tokens, context_mapping)

        return question_embeddings, context_embeddings

    def _get_bert_embeddings_for_batch(self, batch_data, evaluation=False):
        batch_size = len(batch_data[0])

        batch_question_embeddings = []
        batch_context_embeddings = []
        mask_good_ex = []
        new_question_lengths = []
        new_context_lengths = []

        for i in range(batch_size):
            question = batch_data[6][i]
            context = batch_data[1][i]
            answer_end_idx = max(batch_data[15][i], batch_data[17][i]) if not evaluation else None

            q_embeddings, c_embeddings = self._get_bert_embeddings(question, context, answer_end_idx)

            if q_embeddings is None:
                mask_good_ex.append(0)
                continue

            new_q_len = q_embeddings.size(0)
            new_c_len = c_embeddings.size(0)
            new_question_lengths.append(new_q_len)
            new_context_lengths.append(new_c_len)
            mask_good_ex.append(1)

            batch_question_embeddings.append(q_embeddings)
            batch_context_embeddings.append(c_embeddings)

        return batch_question_embeddings, batch_context_embeddings, new_question_lengths, new_context_lengths, \
               mask_good_ex

    def _encode_forward(self, prepared_input):
        cat_mask = prepared_input['cat_mask']
        question_mask = prepared_input['question_mask']
        context_mask = prepared_input['context_mask']
        question_len = prepared_input['question_len']
        question_glove = prepared_input['question_glove']
        context_glove = prepared_input['context_glove']
        question_bert = prepared_input['question_bert']
        context_bert = prepared_input['context_bert']
        cat_question_feat = prepared_input['cat_question_feat']
        cat_context_feat = prepared_input['cat_context_feat']

        word_attention_context, word_attention_question = self._word_attention(context_glove, question_glove,
                                                                               question_glove, question_mask[:, :-1],
                                                                               context_glove, context_mask[:, 1:])
        u_node = self._universal_node.repeat(question_glove.size(0), 1, 1)
        cat_question = torch.cat(
                [question_glove, question_bert, cat_question_feat, word_attention_question], dim=2)
        cat_context = torch.cat(
                [context_glove, context_bert, cat_context_feat, word_attention_context], dim=2)
        cat_input = torch.cat([cat_question, u_node, cat_context], dim=1)

        low_level_info = self._low_info_lstm(cat_input, cat_mask)
        high_level_info = self._high_info_lstm(low_level_info, cat_mask)

        lh_cat_info = torch.cat([low_level_info, high_level_info], dim=2)
        full_info = self._full_info_lstm(lh_cat_info, cat_mask)

        deep_cat_how = torch.cat([cat_input, low_level_info, high_level_info], dim=2)

        deep_question_how = deep_cat_how[:, :question_len + 1]
        low_question_info = low_level_info[:, :question_len + 1]
        high_question_info = high_level_info[:, :question_len + 1]
        full_question_info = full_info[:, :question_len + 1]

        deep_context_how = deep_cat_how[:, question_len:]
        low_context_info = low_level_info[:, question_len:]
        high_context_info = high_level_info[:, question_len:]
        full_context_info = full_info[:, question_len:]

        low_attention_context, low_attention_question = self._low_attention(
                deep_context_how, deep_question_how, low_question_info, question_mask, low_context_info, context_mask)
        high_attention_context, high_attention_question = self._high_attention(
                deep_context_how, deep_question_how, high_question_info, question_mask, high_context_info, context_mask)
        full_attention_context, full_attention_question = self._full_attention(
                deep_context_how, deep_question_how, full_question_info, question_mask, full_context_info, context_mask)

        # sum up two attention representations for universal node
        low_attention_context[:, 0] += low_attention_question[:, -1]
        high_attention_context[:, 0] += high_attention_question[:, -1]
        full_attention_context[:, 0] += full_attention_question[:, -1]

        attention_question_how = torch.cat(
                [low_attention_question, high_attention_question, full_attention_question], dim=2)
        attention_context_how = torch.cat(
                [low_attention_context, high_attention_context, full_attention_context], dim=2)
        attention_cat_how = torch.cat(
                [attention_question_how[:, :-1], attention_context_how], dim=1)

        total_cat_how = torch.cat([low_level_info, high_level_info, full_info, attention_cat_how], dim=2)
        fused_cat_how = self._fusion_lstm(total_cat_how, cat_mask)

        fully_fused_cat_how = torch.cat([cat_input, fused_cat_how], dim=2)
        attention_fully_fused_cat_how = self._self_attention(
                fully_fused_cat_how, fully_fused_cat_how, fully_fused_cat_how, cat_mask)

        fully_fused_cat = torch.cat([fused_cat_how, attention_fully_fused_cat_how], dim=2)
        final_representation_cat = self._final_info_lstm(fully_fused_cat, cat_mask)
        final_plaus_representation_cat = self._final_plausible_info_lstm(fully_fused_cat, cat_mask)

        final_representation_question = final_representation_cat[:, :question_len]
        final_representation_context = final_representation_cat[:, question_len:]
        final_plaus_representation_question = final_plaus_representation_cat[:, :question_len]
        final_plaus_representation_context = final_plaus_representation_cat[:, question_len:]

        return final_representation_question, final_representation_context, final_plaus_representation_question, \
               final_plaus_representation_context

    def _decode_forward(
            self, question_info, context_info, question_plaus_info, context_plaus_info, question_mask, context_mask):
        question_summarized = self._question_summarization(question_info, question_mask[:, :-1])
        question_plaus_summarized = self._question_plausible_summarization(question_plaus_info, question_mask[:, :-1])

        logits_s, logits_e = self._answer_pointer(context_info, question_summarized, context_mask)
        logits_plaus_s, logits_plaus_e = self._plausible_answer_pointer(
                context_plaus_info, question_plaus_summarized, context_mask)

        # answer verifier
        alpha = F.softmax(logits_s, dim=1)
        beta = F.softmax(logits_e, dim=1)

        verify_question_summarized = self._question_verifier_summarization(question_info, question_mask[:, :-1])
        context_summarized_start = alpha.unsqueeze(1).bmm(context_info).squeeze(1)
        context_summarized_end = beta.unsqueeze(1).bmm(context_info).squeeze(1)
        universal_node_info = context_info[:, 0]

        verifier_input = torch.cat([verify_question_summarized, universal_node_info,
                                    context_summarized_start, context_summarized_end], dim=1)
        logits_answerable = self._answer_verifier(verifier_input)

        return logits_s, logits_e, logits_plaus_s, logits_plaus_e, logits_answerable

    def compute_loss(
            self, logits_s, logits_e, logits_plaus_s, logits_plaus_e, logits_answerable, start_idx, end_idx,
            plaus_start_idx, plaus_end_idx, has_answer):
        loss_answer = F.cross_entropy(logits_s, start_idx) + F.cross_entropy(logits_e, end_idx)
        loss_plausible_answer = F.cross_entropy(
                logits_plaus_s, plaus_start_idx) + F.cross_entropy(logits_plaus_e, plaus_end_idx)
        loss_answer_verifier = F.cross_entropy(logits_answerable, has_answer)

        total_loss = loss_answer + loss_plausible_answer + loss_answer_verifier
        return total_loss

    def forward(self, prepared_input):
        encoded_features = self._encode_forward(prepared_input)
        decoded_logits = self._decode_forward(
                *encoded_features, prepared_input['question_mask'],
                prepared_input['context_mask'])
        return decoded_logits

    def _compute_mask(self, x):
        mask = torch.eq(x, 0)
        if self.use_cuda:
            mask = mask.cuda()
        return mask

    def reset_fixed_embeddings(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self._glove_embeddings.weight.size(0):
                self._glove_embeddings.weight.data[offset:] = self._glove_fixed_embeddings
