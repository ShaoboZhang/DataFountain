import numpy as np
import pandas as pd
import torch, os
from sklearn import utils
from tqdm import tqdm
from model.model import labels
from time import sleep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
types = ['高风险'] * 2 + ['中风险'] * 2 + ['低风险'] * 3 + ['可公开'] * 3


def preprocess(train_file_path, dev_file_path):
    dev_df = pd.read_csv(dev_file_path, index_col=0)
    # 从未标注数据中提取新类别数据
    print("数据抽取中...")
    sleep(0.5)
    pseud_df = pd.DataFrame(columns=['class_label', 'content'])
    for i in tqdm(range(len(dev_df))):
        content = dev_df['content'][i]
        if "游戏" in content:
            pseud_df = pseud_df.append({"class_label": "游戏", "content": content}, ignore_index=True)
        elif "娱乐" in content:
            pseud_df = pseud_df.append({"class_label": "娱乐", "content": content}, ignore_index=True)
        elif "体育" in content:
            pseud_df = pseud_df.append({"class_label": "体育", "content": content}, ignore_index=True)
    pseud_game = pseud_df[pseud_df['class_label'] == "游戏"][:1240]
    pseud_sport = pseud_df[pseud_df['class_label'] == "体育"][:1000]
    pseud_enter = pseud_df[pseud_df['class_label'] == "娱乐"][:1000]
    # 与训练集数据合并
    train_df = pd.read_csv(train_file_path, index_col=0)
    pseud_label_df = pd.concat([train_df, pseud_game, pseud_enter, pseud_sport], axis=0)
    pseud_label_df = pseud_label_df.reset_index().drop('index', axis=1)
    pseud_label_df.to_csv("././files/psedu_data.csv", index_label='id')
    print("数据合并完成！")


def process_data(file_path, label2idx, with_label=False):
    df = pd.read_csv(file_path, encoding='utf-8', index_col='id')
    if with_label:
        df = df.replace({'class_label': label2idx})
    return df


def train_test_split(data_df, test_size=0.25, shuffle=True):
    if shuffle:
        data_df = utils.shuffle(data_df)
    idx = int(len(data_df) * test_size)
    train_df = data_df[idx:].reset_index(drop=True)
    test_df = data_df[:idx].reset_index(drop=True)
    return train_df, test_df


def train_eval(model, train_data, val_data, criterion, optim, num_epoch, model_path):
    def evaluate():
        model.eval()
        acc = batch_num = batch_loss = 0.0
        with torch.no_grad():
            for batch_data in val_data:
                batch_data = [data.to(device) for data in batch_data]
                logits = model(batch_data[0], batch_data[1], batch_data[2])
                loss = criterion(logits, batch_data[3])
                acc += accuracy(logits.cpu().detach().numpy(), batch_data[3].cpu().numpy()) * len(batch_data)
                batch_loss += loss.item() * len(batch_data)
                batch_num += len(batch_data)
        model.train()
        return batch_loss / batch_num, acc / batch_num

    # 创建保存模型的目录
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model = model.to(device)
    model.train()
    best_val_loss = float('inf')
    print("开始训练...")
    sleep(0.5)
    torch.cuda.empty_cache()

    for epoch in range(num_epoch):
        with tqdm(train_data, ncols=100) as batch_progress:
            batch_progress.set_description_str(f'Epoch: {epoch + 1}')
            for i, batch_data in enumerate(train_data):
                batch_data = [data.to(device) for data in batch_data]
                logits = model(batch_data[0], batch_data[1], batch_data[2])
                loss = criterion(logits, batch_data[3])
                # backward
                optim.zero_grad()
                loss.backward()
                optim.step()
                # update batch progressor
                train_acc = accuracy(logits.cpu().detach().numpy(), batch_data[3].cpu().numpy())
                if (i + 1) % 30 == 0:
                    val_loss, val_acc = evaluate()
                    # save model if needed
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), model_path)
                else:
                    batch_progress.set_postfix(train_loss=loss.item(), train_acc=train_acc)
                batch_progress.update()
            batch_progress.set_postfix(train_loss=loss.item(), train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            batch_progress.update()


def accuracy(logits, targets):
    preds = np.argmax(logits, axis=-1)
    acc = np.sum(np.equal(preds, targets))
    return acc / len(targets)


def predict(model, model_path, test_data, save_path='result.csv'):
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    pred_idx = []
    pred_labels = []
    pred_types = []
    print("开始预测...")
    sleep(0.5)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_data in tqdm(test_data):
            batch_data = [data.to(device) for data in batch_data]
            logits = model(batch_data[0], batch_data[1], batch_data[2])
            preds = np.argmax(logits.cpu().detach(), axis=1)
            pred_idx += preds.tolist()
            pred_labels += [labels[idx] for idx in preds]
            pred_types += [types[idx] for idx in preds]
    if save_path.endswith('.csv'):
        result = np.array([pred_labels, pred_types]).T
        result = pd.DataFrame(result)
        result.to_csv(save_path, index_label='id', header=['class_label', 'rank_label'])
    elif save_path.endswith('.txt'):
        with open(save_path, 'w') as f:
            for idx in pred_idx:
                f.write(str(idx) + '\n')
    print('数据保存成功!')
