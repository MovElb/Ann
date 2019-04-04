import torch
import torch.nn as nn
from layers import FullAttention, StackedBRNN, Summarize


class BertyNet(nn.Module):
    def __init__(self, opt):
        super(BertyNet, self).__init__()
        self._use_cuda = opt['use_cuda']

        self._build_model(opt)

    def _build_model(self, opt):
        GLOVE_FEAT_DIM = opt['glove_dim']
        BERT_FEAT_DIM = opt['bert_dim']
        POS_DIM = opt['pos_dim']
        POS_SIZE = opt['pos_size']
        NER_DIM = opt['ner_dim']
        NER_SIZE = opt['ner_size']
        CUSTOM_FEAT_DIM = 4
        TOTAL_DIM = GLOVE_FEAT_DIM + BERT_FEAT_DIM + POS_DIM + NER_DIM + CUSTOM_FEAT_DIM

        RNN_HIDDEN_SIZE = opt['rnn_hidden_size']
        ATTENTION_HIDDEN_SIZE = opt['attention_hidden_size']
        DROPOUT_RATE = opt['dropout_rate']

        # TODO : create embedding matrix for glove
        # TODO: get Bert embeddings (not trivial)

        self._pos_embeddings = nn.Embedding(POS_SIZE, POS_DIM, padding_idx=0)
        self._ner_embeddings = nn.Embedding(NER_SIZE, NER_DIM, padding_idx=0)
        self._universal_node = nn.Parameter(torch.zeros(1, TOTAL_DIM))
        nn.init.xavier_normal_(self.universal_node)

        cur_input_size = TOTAL_DIM
        self._low_info_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._high_info_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * (2 * RNN_HIDDEN_SIZE)
        self._full_info_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        attention_input_size = 3 * (2 * RNN_HIDDEN_SIZE)
        self._low_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self._use_cuda)
        self._high_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self._use_cuda)
        self._full_attention = FullAttention(
            attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self._use_cuda)

        cur_input_size = 6 * (2 * RNN_HIDDEN_SIZE)
        self._fusion_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        self_attention_input_size = 2 * RNN_HIDDEN_SIZE + TOTAL_DIM
        self._self_attention = FullAttention(
            self_attention_input_size, ATTENTION_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_cuda=self._use_cuda)

        cur_input_size = self_attention_input_size + 2 * RNN_HIDDEN_SIZE
        self._final_info_lstm = StackedBRNN(
            cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)

        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._question_summarization = Summarize(cur_input_size, dropout_rate=DROPOUT_RATE, use_cuda=self._use_cuda)

    def prepare_input(self, batch_data):
        """Converts token ids to embeddings. Injects universal node into the batch data between question and context.

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

        final_representation_question = final_representation_cat[:, :question_len]
        final_representation_context = final_representation_cat[:, question_len:]

        return final_representation_question, final_representation_context
        # don't forget about plausible answer pointer!

    def _decode_forward(self, question_info, context_info, question_mask, context_mask):
        pass

    def forward(self, batch_data):
        prepared_input = self.prepare_input(batch_data)
        # ... = self.encode_forward(prepared_input)
        pass

    def _compute_mask(self, x):
        mask = torch.eq(x, 0)
        if self._use_cuda:
            mask = mask.cuda()
        return mask