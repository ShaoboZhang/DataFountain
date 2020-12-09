from model import config
from model.model import *
from model.data_utils import preprocess, process_data, train_test_split, train_eval, predict
import torch.utils.data as Data
from torch.nn import CrossEntropyLoss
from transformers import AdamW, logging
from time import time
import warnings, torch


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        func(*args, **kwargs)
        t1 = time()
        print(f"Processing time: {(t1 - t0) / 60:.2f}min")

    return wrapper


@timeit
def main(is_psedu=False):
    logging.set_verbosity_error()
    warnings.filterwarnings('ignore')
    # Preprocess data if needed
    if is_psedu:
        preprocess(config.train_data_path, config.valid_data_path)
    # Load dataset
    train_df = process_data(config.psedu_data_path, label2idx, with_label=True)
    train_df, val_df = train_test_split(train_df)

    # Generate train dataloader
    model_name = config.model_name
    train_dataset = MyDataset(train_df, config.max_len, model_name, with_label=True)
    train_data = Data.DataLoader(train_dataset, batch_size=config.batch_size)
    # Generate validation dataloader
    val_dataset = MyDataset(val_df, config.max_len, model_name, with_label=True)
    val_data = Data.DataLoader(val_dataset, batch_size=config.batch_size, drop_last=True)

    # Load model
    model = MyModel(model_name, config.output_size, config.hidden_size, config.dropout)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.decay)

    # Training
    train_eval(model, train_data, val_data, criterion, optimizer, config.num_epochs, config.model_path)


def gen_result():
    warnings.filterwarnings('ignore')
    test_df = process_data(config.test_data_path, label2idx, with_label=False)
    test_dataset = MyDataset(test_df, config.max_len, config.model_name)
    test_data = Data.DataLoader(test_dataset, batch_size=config.batch_size)
    model = MyModel(config.model_name, config.output_size, config.hidden_size, config.dropout)
    predict(model, config.model_path, test_data, config.output_path)


if __name__ == '__main__':
    # main(is_psedu=False)
    gen_result()
