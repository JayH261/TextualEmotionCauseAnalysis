# Textual Emotion Cause Analysis

This repo contains the code of TASLP'2021 paper:

A Unified Target-Oriented Sequence-to-Sequence Model for Emotion-Cause Pair Extraction

## Original dataset
- Dataset is from WASSA 2021 paper: An End-to-End Network for Emotion-Cause Pair Extraction, {[pdf](https://aaditya-singh.github.io/data/ECPE.pdf)}
- Dataset text file is in text_train folder

## Requirements
- Python 3.6
- PyTorch 1.8.0
- transformers 2.8.0
- pytorch_pretrained_bert 0.6.2
- Download config.json, flax_model.msgpack, pytorch_model.bin, tf_model.h5, vocab.txt of pretrained [BERT-base-cased](https://huggingface.co/bert-base-cased/tree/main) model and put those downloaded files into the empty bert-base-cased folder in utils folder

## Run the model
- Complex UTOS model with our modification
```bash
python main.py
```
- Simplified UTOS model version
```bash
python main2.py
```

## File Description
- Train and test json data files are in folder "all"
- Folder "network" contains model files
    - simple_model.py: UTOS sequence-to-sequence model with BERT encoder and GRU layers for the encoder and decoder
    - modified_model.py: our edited UTOS model
- data_loader.py originally from UTOS authors but we did some modifications because our data and authors data are different
- preprocess_data jupyter notebook is from us, we created this file to convert English dataset text file to JSON format
- Folder result contains some of our results after we trained the model under various parameters
    - To view the result file, use view_result.py and change the file path in that file