import torch
import torch.nn as nn
import torch.nn.functional as F


def variational_dropout(x, dropout_rate=0, training=False, use_cuda=False):
    """
    Args:
        x (Tensor): batch * len * input_size
    """
    if not training or dropout_rate == 0:
        return x
    dropout_mask = 1.0 / (1 - dropout_rate) * torch.bernoulli(x.data.new_zeros(x.size(0), x.size(2)) + 1)
    if use_cuda:
        dropout_mask = dropout_mask.cuda()
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, dropout_rate=0, training=False, dropout_type='alpha', use_cuda=False):
    """
    Args:
        x (Tensor): (batch * len * input_size) or (any other shape)
    """
    if dropout_type not in ('variational', 'simple', 'alpha'):
        raise ValueError('Unknown dropout type = {}'.format(dropout_type))
    if dropout_rate > 0:
        # if x is (batch * len * input_size)
        if dropout_type == 'variational' and len(x.size()) == 3:
            return variational_dropout(x, dropout_rate=dropout_rate, training=training, use_cuda=use_cuda)
        elif dropout_type == 'simple':
            return F.dropout(x, p=dropout_rate, training=training)
        else:
            return F.alpha_dropout(x, p=dropout_rate, training=training)
    else:
        return x


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_type='alpha', dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self._rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self._rnns.append(rnn_type(input_size, hidden_size,
                                       num_layers=1,
                                       bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
                # rnn_input = dropout(rnn_input, dropout_rate=self.dropout_rate,
                #                     training=self.training, dropout_type=self.dropout_type)
            # Forward
            rnn_output = self._rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers

        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
                # dropout_input = dropout(rnn_input, dropout_rate=self.dropout_rate,
                #                         training=self.training, dropout_type=self.dropout_type)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self._rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, padding], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class FullAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0, dropout_type='alpha', use_cuda=True):
        super(FullAttention, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self._projection = nn.Linear(input_size, hidden_size, bias=False)
        self._scaling = nn.Parameter(torch.ones(
                1, hidden_size), requires_grad=True)
        with torch.no_grad():
            nn.init.xavier_normal_(self._projection.weight)

    def forward(self, queries, keys, values_1, values_1_mask, values_2=None, values_2_mask=None):
        dropped_queries = dropout(
                queries, self.dropout_rate, self.training, self.dropout_type, use_cuda=self.use_cuda)
        dropped_keys = dropout(
                keys, self.dropout_rate, self.training, self.dropout_type, use_cuda=self.use_cuda)

        projected_queries = F.relu(self._projection(dropped_queries))
        projected_keys = F.relu(self._projection(dropped_keys))
        scaling_factor = self._scaling.expand_as(projected_keys)

        projected_keys = projected_keys * scaling_factor
        scores = projected_queries.bmm(projected_keys.transpose(2, 1))

        if values_2 is not None:
            scores_T = scores.clone().transpose(2, 1)
            v_mask_2 = values_2_mask.unsqueeze(1).repeat(1, keys.size(1), 1)
            scores_T.masked_fill_(v_mask_2, float('-inf'))
            alpha_2 = F.softmax(scores_T, dim=2)
            output_2 = torch.bmm(alpha_2, values_2)

        v_mask_1 = values_1_mask.unsqueeze(1).repeat(1, queries.size(1), 1)
        scores.masked_fill_(v_mask_1, float('-inf'))
        alpha_1 = F.softmax(scores, dim=2)
        output = torch.bmm(alpha_1, values_1)

        if values_2 is not None:
            return output, output_2
        else:
            return output


class Summarize(nn.Module):
    def __init__(self, input_size, dropout_rate=0, dropout_type='alpha', use_cuda=True):
        super(Summarize, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self.W = nn.Linear(input_size, 1)
        with torch.no_grad():
            nn.init.xavier_normal_(self.W.weight)

    def forward(self, x, mask):
        dropped_x = dropout(x, self.dropout_rate, self.training,
                            self.dropout_type, use_cuda=self.use_cuda)
        coefs = self.W(dropped_x).squeeze(2)
        coefs.masked_fill_(mask, float('-inf'))
        coefs = F.softmax(coefs, 1)
        output = torch.bmm(coefs.unsqueeze(1), x).squeeze(1)

        return output


class PointerNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0, dropout_type='alpha', use_cuda=True):
        super(PointerNet, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self._start_linear = nn.Linear(input_size, input_size)
        self._end_linear = nn.Linear(input_size, input_size)
        with torch.no_grad():
            nn.init.xavier_normal_(self._start_linear.weight)
            nn.init.xavier_normal_(self._end_linear.weight)

    def forward(self, x, y, x_mask):
        """
        x : batch * len * h1
        y : batch * h2
        x_mask : batch * len
        """
        dropped_y = dropout(y, self.dropout_rate, self.training, self.dropout_type, self.use_cuda)

        Wy_s = self._start_linear(dropped_y)
        xWy_s = x.bmm(Wy_s.unsqueeze(2)).squeeze(2)
        xWy_s.masked_fill_(x_mask, float('-inf'))

        Wy_e = self._end_linear(dropped_y)
        xWy_e = x.bmm(Wy_e.unsqueeze(2)).squeeze(2)
        xWy_e.masked_fill_(x_mask, float('-inf'))

        return xWy_s, xWy_e
