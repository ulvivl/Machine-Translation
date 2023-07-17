from enum import Enum
from pathlib import Path
import re

from tokenizers import Tokenizer
from torch.utils.data import Dataset
import torch
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    out_file = open(output_path, 'w+')
    in_file = open(input_path, "r")
    for line in in_file.readlines():
        line = line.strip()
        if line[0] == '<':
            continue
        else:
            out_file.write(line)
            out_file.write('\n')
    return

# process_training_file('data/train.tags.de-en.de', 'aa.txt')
# process_training_file('data/train.tags.de-en.en', 'aa2.txt')

def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    out_file = open(output_path, 'w+')
    in_file = open(input_path, "r")
    for line in in_file.readlines():
        line = line.strip()
        if '<seg' in line.strip():
            line = re.sub('<(s|\/s)eg[^>]*>', '', line).strip()
            out_file.write(line)
            out_file.write('\n')
        else:
            continue
    return

def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """
    # TODO что я тут должна была сделать???
    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file_path,
        tgt_file_path,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        self.train_src = []
        self.train_tgt =  []

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        in_file_src = open(src_file_path, "r")
        in_file_tgt = open(tgt_file_path, "r")
        

        deutch = in_file_src.readlines()
        english = in_file_tgt.readlines()

        for i in range(0, len(english)):
            en_str = english[i].strip()
            de_str = deutch[i].strip()
            tokenized_de = self.src_tokenizer.encode(de_str).ids
            tokenized_en = self.tgt_tokenizer.encode(en_str).ids
            if len(tokenized_de) >= max_len or len(tokenized_en) >= max_len:
                continue
            else:
                self.train_src.append(tokenized_de)
                self.train_tgt.append(tokenized_en)


    def __len__(self):
        return len(self.train_src)

    def __getitem__(self, i):
        return torch.Tensor(self.train_src[i]).long(), torch.Tensor(self.train_tgt[i]).long()


    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """

        pad_idx1 = self.src_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        pad_idx2 = self.tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        src = []
        tgt = []
        for elem in batch:
            src.append(elem[0])
            tgt.append(elem[1])
        src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_idx1)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_idx2)
        return src_padded, tgt_padded


class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    files_en = [str(base_dir) + x.name for x in base_dir.iterdir() if ((not x.is_dir()) and ('en' in x.name))]
    files_de = [str(base_dir) + x.name for x in base_dir.iterdir() if ((not x.is_dir()) and ('de' in x.name))]

    tokenizer_en = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
    tokenizer_en.pre_tokenizer = Whitespace()
    trainer_en = BpeTrainer(vocab_size=30000, special_tokens=[SpecialTokens.UNKNOWN.value, SpecialTokens.PADDING.value, SpecialTokens.BEGINNING.value, SpecialTokens.END.value])
    tokenizer_en.train(files=files_en, trainer=trainer_en)

    tokenizer_de = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
    tokenizer_de.pre_tokenizer = Whitespace()
    trainer_de = BpeTrainer(vocab_size=30000, special_tokens=[SpecialTokens.UNKNOWN.value, SpecialTokens.PADDING.value, SpecialTokens.BEGINNING.value, SpecialTokens.END.value])
    tokenizer_de.train(files=files_de, trainer=trainer_de)

    tokenizer_en.post_processor = TemplateProcessing(
    single=f"{SpecialTokens.BEGINNING.value} $A {SpecialTokens.END.value}",
    special_tokens=[
        (SpecialTokens.BEGINNING.value, tokenizer_en.token_to_id(SpecialTokens.BEGINNING.value)),
        (SpecialTokens.END.value, tokenizer_en.token_to_id(SpecialTokens.END.value)),
        ],
    )

    tokenizer_de.post_processor = TemplateProcessing(
    single=f"{SpecialTokens.BEGINNING.value} $A {SpecialTokens.END.value}",
    special_tokens=[
        (SpecialTokens.BEGINNING.value, tokenizer_de.token_to_id(SpecialTokens.BEGINNING.value)),
        (SpecialTokens.END.value, tokenizer_de.token_to_id(SpecialTokens.END.value)),
        ],
    )

    tokenizer_de.save(str(save_dir / 'tokenizer_de.json'))
    tokenizer_en.save(str(save_dir / 'tokenizer_en.json'))