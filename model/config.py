#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Shaobo
@Discribe: Configuration parameters
@FilePath: /Workshop/model/config.py
"""

# data
train_data_path = './files/labeled_data.csv'
valid_data_path = './files/unlabeled_data.csv'
test_data_path = './files/test_data.csv'
psedu_data_path = './files/psedu_data.csv'

max_len: int = 300  # exclusive of BOS and EOS

# model
# model_name = './lib/hfl/'
# model_name = './lib/clue/'
# model_name = './lib/albert/'
model_name = './lib/bert/'
embed_size = 156
output_size = 768
hidden_size = 128
dropout = 0.4

# train/test
batch_size = 64
num_epochs = 10
lr = 2e-5
decay = 1e-2

# model save
model_path = './save/basic_bert1.pt'
# output_path = './result/result.txt'
output_path = './result/result.csv'
