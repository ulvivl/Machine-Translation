import torch
import torch.nn as nn
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from data import SpecialTokens


from model import TranslationModel

detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()



def _generate_square_subsequent_mask(seq_len):        
        mask = torch.ones((seq_len, seq_len)) * -torch.inf
        mask = torch.triu(mask, diagonal=1)
        return mask

def _greedy_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """
    start_symbol = tgt_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
    end_symbol = tgt_tokenizer.token_to_id(SpecialTokens.END.value)

    src_mask = torch.zeros((src.shape[-1], src.shape[-1]), device=device).type(torch.bool)
    memory = model.encode(src, src_mask).to(device)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_mask = (_generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol:
            break
    return ys


def _beam_search_decode_not_batched(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    start_symbol = tgt_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
    end_symbol = tgt_tokenizer.token_to_id(SpecialTokens.END.value)

    src_mask = torch.zeros((src.shape[-1], src.shape[-1]), device=device).type(torch.bool)
    memory = model.encode(src, src_mask).to(device)
    memory = memory.repeat(beam_size, 1, 1)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    tgt_mask = (_generate_square_subsequent_mask(ys.size(-1)).type(torch.bool)).to(device)    
    out = model.decode(ys, memory[0].unsqueeze(0), tgt_mask)
    prob = nn.Softmax(dim=-1)(model.generator(out[:, -1]))
    prob_now = prob

    max_val, max_ind = torch.sort(prob_now.flatten(), descending=True)
    max_val = max_val[:beam_size]
    max_ind = max_ind[:beam_size]

    path_prob = max_val.reshape((max_val.shape[0], 1))

    ys = torch.ones(beam_size, 1).fill_(start_symbol).type(torch.long).to(device)
    ys = torch.cat((ys, max_ind.reshape(max_ind.shape[0], 1)), dim=-1)


    for i in range(max_len - 2):
        tgt_mask = (_generate_square_subsequent_mask(ys.size(-1)).type(torch.bool)).to(device)    
        out = model.decode(ys, memory, tgt_mask)
        prob = nn.Softmax(dim=-1)(model.generator(out[:, -1]))


        prob_now = prob * path_prob
        b, c = prob_now.shape
        max_val, max_ind = torch.topk(prob_now.flatten(), beam_size)
        batch_ind = (max_ind // c) % b
        col_ind = max_ind % c

        unsort_max = torch.sort(max_ind).indices
        col_ind = col_ind[unsort_max]
        max_val = max_val[unsort_max]
        batch_ind = batch_ind[unsort_max]

        path_prob = max_val.reshape((max_val.shape[0], 1))
        ys = torch.cat((ys[batch_ind, :], col_ind.reshape(col_ind.shape[0], 1)), dim=-1)
        if (col_ind == end_symbol).all():
            break
    ans = ys[torch.argmax(path_prob)]
    index_first_eos = (ans == end_symbol).nonzero()
    if len(index_first_eos) != 0:
        index_first_eos = index_first_eos[0].item()
        ans = ans[: index_first_eos + 1]
    return ans



def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    pass


@torch.inference_mode()
def translate(
    model: torch.nn.Module,
    src_sentences,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    translation_mode: str,
    device: torch.device,
):
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """
    tokens = src_tokenizer.encode(src_sentences).ids
    if translation_mode == 'greedy':
        tokenized_translation = _greedy_decode(model, torch.Tensor(tokens).to(device), len(tokens) + 5, tgt_tokenizer, device).flatten()
    if translation_mode == 'beam':
        tokenized_translation = _beam_search_decode_not_batched(model, torch.Tensor(tokens).to(device), len(tokens) + 5, tgt_tokenizer, device, 5).flatten()
    unnorm = tgt_tokenizer.encode(
                    tgt_tokenizer.decode(tokenized_translation.long().cpu().tolist(), skip_special_tokens=True),
                     add_special_tokens=False).tokens
    text = mpn.normalize(detok.detokenize(unnorm))
    return text



