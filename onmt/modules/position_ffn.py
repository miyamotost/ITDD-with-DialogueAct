"""
Position feed-forward network from "Attention is All You Need"
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class PositionwiseFeedForward2(nn.Module):

    def __init__(self, d_model_in, d_model_out, d_ff, dropout=0.1, model_ffn_mode='resnet_nLN'):
        super(PositionwiseFeedForward2, self).__init__()
        self.model_ffn_mode = model_ffn_mode
        self.w_1 = nn.Linear(d_model_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model_out)
        self.layer_norm = nn.LayerNorm(d_model_in, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.w_x = nn.Linear(d_model_in, d_model_out)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        if self.model_ffn_mode == 'resnet_LN':
            x = self.layer_norm(x)
        x = self.w_x(x)

        return output + x
