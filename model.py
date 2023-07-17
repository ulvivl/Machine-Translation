import torch.nn as nn
from torch import Tensor
import torch
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TranslationModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        dim_feedforward: int,
        n_head: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout_prob: float,
        max_len: int, 
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                nhead=n_head,
                                num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_decoder_layers,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout_prob, batch_first=True)

        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_embed = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, max_len)


    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        emb_src = self.positional_encoding(self.src_embed(src_tokens))
        emb_tgt = self.positional_encoding(self.tgt_embed(tgt_tokens))
        outs = self.transformer(emb_src, emb_tgt, None, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask)
        
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        return self.generator(outs)
    
    def encode(self, src: Tensor, src_mask: Tensor):
            return self.transformer.encoder(self.positional_encoding(
                            self.src_embed(src.long())), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_embed(tgt.long())), memory,
                          tgt_mask)


class PositionalEncoding(nn.Module):
    
    def __init__(self, emb_size, max_len=512):
        super().__init__()

        pos_indexes = torch.arange(max_len).unsqueeze(1) # k
        
        even_indexes = torch.arange(0, emb_size, 2) # 2i
        
        # считаем exp(log(1 / n^{2i / d}))
        denominator = torch.exp(even_indexes * (-math.log(10000.0) / emb_size))
        
        # pos encoding empty (batch_size x seq_len x embed size)
        self.pe = torch.zeros(1, max_len, emb_size)
        
        # fill even indexes of pos encoding with sin
        self.pe[0, :, 0::2] = torch.sin(pos_indexes * denominator)
        
        # fill odd indexes of pos encoding with cos
        self.pe[0,:, 1::2] = torch.cos(pos_indexes * denominator)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x_bs, x_seq_len, x_emb_size = x.shape
        x = x.to(DEVICE)
        self.pe = self.pe.to(DEVICE)
        x = x + self.pe[:, :x_seq_len, :]
        return x