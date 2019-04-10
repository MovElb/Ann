import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FullAttention, StackedBRNN, Summarize, PointerNet
from pytorch_pretrained_bert import BertModel, BertTokenizer


class BertyNet(nn.Module):
    def __init__(self, opt, glove_embeddings=None):
        """
            glove_embeddings {list of lists} -- matrix of Glove embeddings with shape (vocab_len, embedding_dim) or None
        """
        super(BertyNet, self).__init__()
        self.use_cuda = opt['use_cuda']
        self.opt = opt
        self.g_pretr_embeddings = glove_embeddings
        self._build_model()

    def _build_model(self):
        GLOVE_FEAT_DIM = self.opt['glove_dim']
        BERT_FEAT_DIM = self.opt['bert_dim']
        POS_DIM = self.opt['pos_dim']
        POS_SIZE = self.opt['pos_size']
        NER_DIM = self.opt['ner_dim']
        NER_SIZE = self.opt['ner_size']
        CUSTOM_FEAT_DIM = 4
        TOTAL_DIM = GLOVE_FEAT_DIM + BERT_FEAT_DIM + POS_DIM + NER_DIM + CUSTOM_FEAT_DIM

        RNN_HIDDEN_SIZE = self.opt['rnn_hidden_size']
        ATTENTION_HIDDEN_SIZE = self.opt['attention_hidden_size']
        DROPOUT_RATE = self.opt['dropout_rate']

        # TODO: get Bert embeddings (not trivial)

        if self.g_pretr_embeddings is not None:
            glove_pretr_embeddings = torch.tensor(self.g_pretr_embeddings)
            self._glove_embeddings = nn.Embedding(
                glove_pretr_embeddings.size(0),
                glove_pretr_embeddings.size(1),
                padding_idx=0)
            self._glove_embeddings.weight[2:, :] = glove_pretr_embeddings[2:, :]
            if self.opt['tune_partial'] > 0:
                assert self.opt['tune_partial'] + 2 < glove_pretr_embeddings.size(0)
                glove_fixed_embeddings = glove_pretr_embeddings[self.opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embeddings', glove_fixed_embeddings)
                self._glove_fixed_embeddings = glove_fixed_embeddings

        else:
            self._glove_embeddings = nn.Embedding(self.opt['vocab_size'], GLOVE_FEAT_DIM, padding_idx=0)

        self._pos_embeddings = nn.Embedding(POS_SIZE, POS_DIM, padding_idx=0)
        self._ner_embeddings = nn.Embedding(NER_SIZE, NER_DIM, padding_idx=0)
        self._universal_node = nn.Parameter(torch.zeros(1, TOTAL_DIM))
        nn.init.xavier_normal_(self._universal_node)

        cur_input_size = TOTAL_DIM
        self._low_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._high_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * (2 * RNN_HIDDEN_SIZE)
        self._full_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        attention_input_size = 3 * (2 * RNN_HIDDEN_SIZE)
        self._low_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._high_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)
        self._full_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self.use_cuda)

        cur_input_size = 6 * (2 * RNN_HIDDEN_SIZE)
        self._fusion_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        self_attention_input_size = 2 * RNN_HIDDEN_SIZE + TOTAL_DIM
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

        cur_input_size = 3 * (2 * RNN_HIDDEN_SIZE)
        self._answer_verifier = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(cur_input_size, 2))

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if self.use_cuda:
            self.bert_model.cuda()
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_input(self, batch_data, evaluation=False):
        """Converts token ids to embeddings. Injects universal node into the batch data between question and context.
        Note that we need to increment ys, ye by 1 (since we inserted universal node). But leave unanswerable questions
        with ys, ye = (0, 0). Maybe we should do that in batch generator ?
        Arguments:
            batch_data....

        Returns:
            dict of Tensor with following values:
                ['cat_input'] - Tensor with size (B * L * output_dim), where B - batch size,
                    L - maximum (probably, padded) length of "question|universal_node|context" sequence,
                    output_dim - dimension of all concatenated features per token.
                ['question_mask']
                ['context_mask
                ['cat_mask']
                ['question_len']
                ['context_len']
        """

        # TODO: extract essential data, concatenate "question|UNODE|context"
        return {}

    def _get_bert_embeddings(self, question_tokens, context_tokens, answer_end_idx):
        MAX_SEQ_LEN = 512
        CLS_IND = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        SEP_IND = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

        def get_wordpiece_tokenization(tokens):
            wp_tokens = []
            orig_to_tok_map = []
            for token in tokens:
                if token not in ['[CLS]', '[SEP]']:
                    orig_to_tok_map.append(len(wp_tokens))
                wp_tokens.extend(self.tokenizer.tokenize(token))
            indexed_wp_tokens = self.tokenizer.convert_tokens_to_ids(wp_tokens)
            return indexed_wp_tokens, orig_to_tok_map

        def truncate_sequence(seq_tokens, mapping, desired_len):
            while (len(seq_tokens) > desired_len):
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
            context_bert_embeddings = sum_last_four_layers[context_mapping]

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
            if answer_end_idx >= len(context_mapping):
                return None, None
            truncate_sequence(question_wp_tokens, question_mapping, MAX_SEQ_LEN - 2)

            question_embeddings = embed_separately(question_wp_tokens, question_mapping)
            context_embeddings = embed_separately(context_wp_tokens, context_mapping)

        return question_embeddings, context_embeddings

    def _get_bert_embeddings(self, batch_data):
        BERT_HID_SIZE = 768

        max_question_len = batch_data['question_len']
        max_context_len = batch_data['context_len']
        batch_size = len(batch_data['question_tokens'])

        batch_question_embeddings = []
        batch_context_embeddings = []
        leaved_example_ids = []
        new_question_lengths = []
        new_context_lengths = []

        for i in range(batch_size):
            question = batch_data['question_tokens'][i]
            context = batch_data['context_tokens'][i]
            answer_end_idx = batch_data['answer_end'][i]

            q_embeddings, c_embeddings = self._get_bert_embeddings(question, context, answer_end_idx)

            if q_embeddings is None:
                continue

            new_q_len = q_embeddings.size(0)
            new_c_len = c_embeddings.size(0)
            new_question_lengths.append(new_q_len)
            new_context_lengths.append(new_c_len)
            leaved_example_ids.append(i)

            q_padded_embeddings = torch.zeros(max_question_len, BERT_HID_SIZE)
            c_padded_embeddings = torch.zeros(max_context_len, BERT_HID_SIZE)

            if self.use_cuda:
                q_padded_embeddings = q_padded_embeddings.cuda()
                c_padded_embeddings = c_padded_embeddings.cuda()

            q_padded_embeddings[max_question_len - new_q_len:] = q_embeddings
            c_padded_embeddings[:new_c_len] = c_embeddings

            batch_question_embeddings.append(q_padded_embeddings.unsqueeze(0))
            batch_context_embeddings.append(c_padded_embeddings.unsqueeze(0))

        batch_question_embeddings = torch.cat(batch_question_embeddings, dim=0)
        batch_context_embeddings = torch.cat(batch_context_embeddings, dim=0)

        return batch_question_embeddings, batch_context_embeddings, new_question_lengths, new_context_lengths, \
               leaved_example_ids

    def _encode_forward(self, prepared_input):
        cat_input = prepared_input['cat_input']
        cat_mask = prepared_input['cat_mask']
        question_mask = prepared_input['question_mask']
        context_mask = prepared_input['context_mask']
        question_len = prepared_input['question_len']

        low_level_info = self._low_info_lstm(cat_input, cat_mask)
        high_level_info = self._high_info_lstm(low_level_info, cat_mask)

        lh_cat_info = torch.cat([low_level_info, high_level_info], dim=2)
        full_info = self._full_info_lstm(lh_cat_info, cat_mask)

        deep_cat_how = torch.cat(
            [low_level_info, high_level_info, full_info], dim=2)

        deep_question_how = deep_cat_how[:question_len + 1]
        low_question_info = low_level_info[:question_len + 1]
        high_question_info = high_level_info[:question_len + 1]
        full_question_info = full_info[:question_len + 1]

        deep_context_how = deep_cat_how[question_len:]
        low_context_info = low_level_info[question_len:]
        high_context_info = high_level_info[question_len:]
        full_context_info = full_info[question_len:]

        low_attention_context, low_attention_question = self._low_attention(
            deep_context_how, deep_question_how, low_question_info, question_mask, low_context_info, context_mask)
        high_attention_context, high_attention_question = self._high_attention(
            deep_context_how, deep_question_how, high_question_info, question_mask, high_context_info, context_mask)
        full_attention_context, full_attention_question = self._high_attention(
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

        total_cat_how = torch.cat([deep_cat_how, attention_cat_how], dim=2)
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
                self._glove_embeddings.weight[offset:] = self._glove_fixed_embeddings
