# Machine Translation

The goal is to build a pipeline for machine translation and implement several decoding schemes.

## Data 
You can find data for all of our experiments in the directory `data`. The data was obtained from [IWSLT 2017 German to English](https://wit3.fbk.eu/2017-01-b) translation dataset. 
You can download the original dataset with `gdown --fuzzy "https://drive.google.com/file/d/12ycYSzLIG253AFN35Y6qoyf9wtkOjakp/view?usp=sharing"`
(make sure to run `pip install gdown` beforehand). 

## Tasks

This assignment has 6 compulsory tasks and two bonus ones:

### The pipeline description
Firstly, the dataset files were parsed into [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) objects. Secondly, a Byte-Pair Encoding vocabulary from training and validation data were built using the [tokenizers](https://github.com/huggingface/tokenizers) library All necessary function for these steps can be found in `data.py`. To run the pipeline you can launch the `process_data.py`. It is also possible to find parsed data and trained tokenizers in folders `processed_data` and `tokenizers`, respectively. Thirdly, the translation model was implemented using the [torch.nn.Transformer (https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer) class (`model.py`). The training procedure can be found in `train_model.py`. To measure the translation quality 
a metric `BLEU` was used. After training the model 2 types of decoding procedures were implemented: greedy decode and beam search decode (all functions can be found in `decoding.py`).

### Training Examples
Simple training procedure can be found in Machine_Translation.ipynb
