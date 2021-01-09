"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
import onmt.opts as opts

from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward, PositionwiseFeedForward2


class STransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(STransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ATransformerEncoderLayer(nn.Module):
    """
    Incremental Transformer

    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, model_mode, model_mode2, model_ffn_mode, d_model, heads, d_ff, dropout):
        super(ATransformerEncoderLayer, self).__init__()

        self.model_mode = model_mode
        self.model_mode2 = model_mode2
        self.model_ffn_mode = model_ffn_mode

        # TODO: branch test
        if self.model_mode2 in ['default']:
            # attention
            self.self_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            self.knowledge_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            self.context_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            # feed forward
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            # layer normalization
            self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
            # debug
            print('default == {}'.format(self.model_mode2))

        elif self.model_mode2 in ['ffn']:
            # attention
            self.self_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            self.knowledge_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            self.context_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
            # feed forward
            if self.model_mode in ['top_act']:
                d_act = 1
                print('top_act == {}'.format(self.model_mode))
            elif self.model_mode in ['all_acts']:
                d_act = 4
                print('all_acts == {}'.format(self.model_mode))
            else:
                print('choose valid option -model_mode')
                exit()
            if self.model_ffn_mode in ['additional']:
                self.feed_forward = PositionwiseFeedForward(d_model+d_act, d_ff, dropout)
                self.feed_forward2 = nn.Linear(d_model+d_act, d_model)
                print('additional == {}'.format(self.model_ffn_mode))
            elif self.model_ffn_mode in ['resnet_nLN', 'resnet_LN']:
                self.feed_forward = PositionwiseFeedForward2(d_model+d_act, d_model, d_ff, dropout, self.model_ffn_mode)
                print('resnet_LN|resnet_nLN == {}'.format(self.model_ffn_mode))
            else:
                print('choose valid option -model_ffn_mode')
                exit()
            # layer normalization
            self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
            # debug
            print('ffn == {}'.format(self.model_mode2))

        elif self.model_mode2 in ['utt_emb']:
            print('utt_emb == {}'.format(self.model_mode2))

        else:
            print('choose valid option -model_mode2')
            exit()

        exit()

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, src_mask, knl_bank, knl_mask, his_bank, his_mask, src_da_label=1):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, mask=src_mask)
        query = self.dropout(query) + inputs
        query_norm = self.layer_norm_2(query)
        knl_out, _ = self.knowledge_attn(knl_bank, knl_bank, query_norm, mask=knl_mask)
        knl_out = self.dropout(knl_out) + query

        if his_bank is not None:
            his_bank = his_bank.transpose(0, 1).contiguous()
            knl_out_norm = self.layer_norm_3(knl_out)
            out, _ = self.context_attn(his_bank, his_bank, knl_out_norm, mask=his_mask)
            out = self.dropout(out) + knl_out
        else:
            out = knl_out

        # TODO: branch test
        if self.model_mode2 in ['default']:
            out = self.feed_forward(out)

        elif self.model_mode2 in ['ffn']:
            if self.model_mode in ['top_act']:
                da_emb = torch.empty(out.shape[0], out.shape[1], 1, device=out.device) # da_emb.shape = torch.Size([13, 50, 1])
                for i in range(out.shape[0]):
                    da_emb[i].fill_(src_da_label[i])
            elif self.model_mode in ['all_acts']:
                da_emb = torch.empty(out.shape[0], out.shape[1], 4, device=out.device) # da_emb.shape = torch.Size([13, 50, 4])
                for i in range(out.shape[0]):
                    da_emb[i, :, 0].fill_(src_da_label[0][i])
                    da_emb[i, :, 1].fill_(src_da_label[1][i])
                    da_emb[i, :, 2].fill_(src_da_label[2][i])
                    da_emb[i, :, 3].fill_(src_da_label[3][i])
            out = torch.cat((out, da_emb), dim=2) # out.shape = torch.Size([13, 50, 513 or 516])
            if self.model_ffn_mode in ['additional']:
                out = self.feed_forward(out) # out.shape = torch.Size([13, 50, 513 or 516])
                out = self.feed_forward2(out)
            elif self.model_ffn_mode in ['resnet_nLN', 'resnet_LN']:
                out = self.feed_forward(out)

        elif self.model_mode2 in ['utt_emb']:
            print('utt_emb == {}'.format(self.model_mode2))
            exit()

        else:
            print('choose valid option -model_mode2')
            exit()

        return out


class KNLTransformerEncoder(EncoderBase):
    """
    Self-Attentive Encoder

    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(KNLTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [STransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), mask


class HTransformerEncoder(EncoderBase):
    """
    Incremental Transformer Encoder

    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, model_mode, model_mode2, model_ffn_mode, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(HTransformerEncoder, self).__init__()
        self.model_mode = model_mode
        self.model_mode2 = model_mode2
        self.model_ffn_mode = model_ffn_mode
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [ATransformerEncoderLayer(model_mode, model_mode2, model_ffn_mode, d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, history_bank=None, knl_bank=None, knl_mask=None, his_mask=None, src_da_label=1):
        """ See :obj:`EncoderBase.forward()`"""

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        padding_idx = self.embeddings.word_padding_idx
        src_mask = words.data.eq(padding_idx).unsqueeze(1)
        # if history_bank is None:
        #     history_bank = torch.randn(out.size()).cuda().transpose(0, 1).contiguous()
        # temp = torch.cat([out, history_bank.transpose(0, 1).contiguous()], 2)
        # out = out + self.w(temp)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, src_mask, knl_bank, knl_mask, history_bank, his_mask, src_da_label)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), src_mask


class TransformerEncoder(EncoderBase):
    def __init__(self, model_mode, model_mode2, model_ffn_mode, num_layers, d_model, heads, d_ff, dropout, embeddings):
        # KTransformerEncoder 与 HTransformerEncoder暂时共享embedding
        super(TransformerEncoder, self).__init__()

        self.model_mode = model_mode
        self.model_mode2 = model_mode2
        self.model_ffn_mode = model_ffn_mode

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.knltransformer = KNLTransformerEncoder(num_layers, d_model, heads, d_ff, dropout, embeddings)
        self.histransformer = KNLTransformerEncoder(num_layers, d_model, heads, d_ff, dropout, embeddings)
        self.htransformer = HTransformerEncoder(model_mode, model_mode2, model_ffn_mode, num_layers, d_model, heads, d_ff, dropout, embeddings)

    def forward(self, src, knl=None, lengths=None, knl_lengths=None, src_da_label=(1, 1, 1)):
        history = self.history2list(src, knl, src_da_label, self.model_mode)
        tgt_knl = knl[600:, :, :]
        # utterance in k and document in k+1 are passed through Self-Attentive Encoder for Decoder
        emb, knl_bank_tgt, knl_mask = self.knltransformer(tgt_knl, None)
        emb, src_bank, src_mask = self.histransformer(src[100:, :, :], None)
        his_bank = None
        his_mask = None
        for h in history:
            # document in k-2, k-1, k are passed through Self-Attentive Encoder
            # then utterance and dialogue act label in k-2, k-1, k and encoded documents
            # are passed through Incremental Transformer Encoder
            u = h[0]
            k = h[1]
            da_label = h[2]
            emb, knl_bank, knl_mask = self.knltransformer(k, None)
            knl_bank_input = knl_bank.transpose(0, 1).contiguous()
            emb, his_bank, his_mask = self.htransformer(u, his_bank, knl_bank_input, knl_mask, his_mask, da_label)
        return emb, his_bank, src_bank, knl_bank_tgt, lengths

    @staticmethod
    def history2list(src, knl, src_da_label, model_mode):
        u1 = src[:50, :, :]
        u2 = src[50:100, :, :]
        u3 = src[100:, :, :]
        k1 = knl[:200, :, :]
        k2 = knl[200:400, :, :]
        k3 = knl[400:600, :, :]
        if model_mode == 'top_act':
            l1 = src_da_label[0]
            l2 = src_da_label[1]
            l3 = src_da_label[2]
        elif model_mode == 'all_acts':
            l1 = (src_da_label[0], src_da_label[1], src_da_label[2], src_da_label[3])
            l2 = (src_da_label[4], src_da_label[5], src_da_label[6], src_da_label[7])
            l3 = (src_da_label[8], src_da_label[9], src_da_label[10], src_da_label[11])
        else:
            l1 = 0
            l2 = 0
            l3 = 0
        return (u1, k1, l1), (u2, k2, l2), (u3, k3, l3)
