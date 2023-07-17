from argparse import ArgumentParser
from pathlib import Path

import wandb

import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange

from data import TranslationDataset
from decoding import translate
from model import TranslationModel
from data import SpecialTokens
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn as nn
import math
import numpy as np
from time import time
import matplotlib.pyplot as plt


s_tokenizer = Tokenizer.from_file(str("tokenizer/tokenizer_de.json"))
t_tokenizer = Tokenizer.from_file(str("tokenizer/tokenizer_en.json"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _generate_square_subsequent_mask(seq_len):        
        mask = torch.ones((seq_len, seq_len)) * -torch.inf
        mask = torch.triu(mask, diagonal=1)
        return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[-1]
    tgt_seq_len = tgt.shape[-1]

    tokenizer_de = Tokenizer.from_file("tokenizer/tokenizer_de.json")
    tokenizer_en = Tokenizer.from_file("tokenizer/tokenizer_en.json")

    pad_idx1 = tokenizer_de.token_to_id(SpecialTokens.PADDING.value)
    pad_idx2 = tokenizer_en.token_to_id(SpecialTokens.PADDING.value)
    

    tgt_mask = _generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)


    src_padding_mask = (src == pad_idx1)
    tgt_padding_mask = (tgt == pad_idx2)
    return tgt_mask, src_padding_mask, tgt_padding_mask

def train_epoch(
    model: TranslationModel,
    train_dataloader,
    optimizer,
    device,
    scheduler,
):
    model.train()
    losses = []
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        tgt_mask, src_padding_mask, tgt_padding_mask = tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

        logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]

        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        optimizer.step()
        if not scheduler is None:
            scheduler.step()
        losses.append(loss.item())

    return losses

    
@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, device):
    model.eval()
    losses = []
    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        tgt_mask, src_padding_mask, tgt_padding_mask = tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

        logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)

        tgt_out = tgt[:, 1:]

        loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses.append(loss.item())
    return losses


def train_model(data_dir, tokenizer_path, num_epochs):

    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )

    # print('Enter run name:')
    # r_name = input()

    # wandb.init(
    #     project="Machine translation (DL)",
    #     name='r_name',
    #     config={
    #         "NUM_EPOCHS" : num_epochs,
    #         "BATCH_SIZE" : 128,
    #         "NUM_WORKERS" : 2,
    #         "LR" : 1e-4,
    #         "MAX_LR" : 1e-3,
    #         "WARMUP_STEPS" : 300,
    #         }) 
    # config = wandb.config
    
    config={
            "NUM_EPOCHS" : num_epochs,
            "BATCH_SIZE" : 128,
            "NUM_WORKERS" : 2,
            "LR" : 1e-4,
            "MAX_LR" : 1e-3,
            "WARMUP_STEPS" : 300,
            }

    train_dataloader = DataLoader(train_dataset,
                                        pin_memory=True,
                                        drop_last=True, 
                                        shuffle=True, 
                                        batch_size=config['BATCH_SIZE'], 
                                        num_workers=config['NUM_WORKERS'],
                                        collate_fn=train_dataset.collate_translation_data)

    val_dataloader = DataLoader(val_dataset, 
                                    pin_memory=True, 
                                    batch_size=config['BATCH_SIZE'], 
                                    num_workers=config['NUM_WORKERS'],
                                    collate_fn=val_dataset.collate_translation_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TranslationModel(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=64,
        dim_feedforward=512,
        n_head=8,
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        dropout_prob=0.1,
        max_len=128
    )
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'], weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **{
        "steps_per_epoch": len(train_dataloader),
        "epochs": num_epochs,
        "anneal_strategy": "cos",
        "max_lr": config['MAX_LR'],
        "pct_start": 0.1
    })

    min_val_loss = float("inf")

    all_loss_train = []
    all_loss_val = []
    all_lrs = []

    for epoch in trange(1, num_epochs + 1):
        break # TODO для обучения убрать!!
        train_loss = train_epoch(model, train_dataloader, optimizer, device, lr_scheduler)
        val_loss = evaluate(model, val_dataloader, device)
        curr_lr = lr_scheduler.get_lr()

        steps = len(train_dataset) / config['BATCH_SIZE']
        all_loss_val.append((steps * (epoch + 1), np.mean(val_loss)))
        all_loss_train.extend(train_loss)
        all_lrs.extend(curr_lr)

        plot_history(all_loss_train, all_loss_val, all_lrs)
        print(f'Epoch: {epoch}\nTrain loss: {np.mean(train_loss)}\t Val loss: {np.mean(val_loss)}\n')

        metrics = {"train/train_loss": train_loss,
                    "lr": curr_lr}
        val_metrics = {"val/val_loss": val_loss}

        wandb.log({**metrics, **val_metrics})

        if np.mean(val_loss) < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = np.mean(val_loss)
            if epoch % 5 == 0:
                file_name = f'checkpoint_best_epoch_{epoch}.pth'
                wandb.save(file_name)

        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("chkpt/checkpoint_best_epoch_20.pth", map_location=DEVICE))
    return model

def plot_history(train_history, val_history, lrs, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(val_history)
    
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    
    plt.legend(loc='best')
    plt.grid()

    plt.show(block=True)

    plt.figure()
    plt.title('lr')
    plt.plot(lrs, label='lr', zorder=0)
    plt.show(block=True)



def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    model.eval()
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))
    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_greedy.txt", "w+"
    ) as output_file:
        # translate with greedy search
        for line in input_file.readlines():
            line = line.strip()
            text_en = translate(model, line, src_tokenizer, tgt_tokenizer, 'greedy', DEVICE)
            greedy_translations.append(text_en)
            output_file.write(text_en)
            output_file.write('\n')

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_beam.txt", "w+"#beam
    ) as output_file:
        for line in input_file.readlines():
            line = line.strip()
            text_en = translate(model, line, src_tokenizer, tgt_tokenizer, 'beam', DEVICE)
            beam_translations.append(text_en)
            output_file.write(text_en)
            output_file.write('\n')

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score
    print(f"\nBLEU with greedy search: {bleu_greedy}")

    bleu = BLEU()
    bleu_beam = bleu.corpus_score(beam_translations, [references]).score
    print(f"BLEU with beam search: {bleu_beam}")

if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()


    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)


