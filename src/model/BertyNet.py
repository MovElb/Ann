import torch
import torch.nn as nn
from layers import StackedBRNN

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
        DROPOUT_RATE = opt['dropout_rate']

        # TODO : create embedding matrix for glove
        # TODO: get Bert embeddings (not trivial)

        self._pos_embeddings = nn.Embedding(POS_SIZE, POS_DIM, padding_idx=0)
        self._ner_embeddings = nn.Embedding(NER_SIZE, NER_DIM, padding_idx=0)
        self._universal_node = nn.Parameter(torch.zeros(1, TOTAL_DIM))
        nn.init.xavier_normal_(self.universal_node)

        cur_input_size = TOTAL_DIM
        self._low_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * RNN_HIDDEN_SIZE
        self._high_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)
        cur_input_size = 2 * (2 * RNN_HIDDEN_SIZE)
        self._full_info_lstm = StackedBRNN(cur_input_size, RNN_HIDDEN_SIZE, 1, dropout_rate=DROPOUT_RATE)


    def prepare_input(self, batch_data):
        """Converts token ids to embeddings. Injects universal node into the batch data between question and context.
        
        Arguments:
            batch_data....
        
        Returns:
            prepared_input - Tensor with size (B * L * output_dim), where B - batch size,
                L - maximum (probably, padded) length of "question|universal_node|context" sequence,
                output_dim - dimension of all concatenated features per token.
            question_mask
            context_mask
            cat_mask
            question_len
            context_len
        """

        # TODO: extract essential data, concatenate "question|UNODE|context"
        return None, None, None, None, None, None

    def encode_forward(self, batch_data):
        input, question_mask, context_mask, cat_mask, question_len, context_len = self.prepare_input(batch_data)
        low_level_info = self._low_info_lstm(input, cat_mask)

        high_level_info = self._high_info_lstm(low_level_info, cat_mask)
        lh_cat_info = torch.cat([low_level_info, high_level_info], dim=2)
        full_info = self._full_info_lstm(lh_cat_info, cat_mask)

        deep_fusion_cat_info = torch.cat([low_level_info, high_level_info, full_info], dim=2)


    def forward(self):
        pass

    def _compute_mask(self, x):
        mask = torch.eq(x, 0)
        if self._use_cuda:
            mask = mask.cuda()
        return mask
