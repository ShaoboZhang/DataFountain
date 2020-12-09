from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel
import torch.utils.data as tud

import torch, fasttext
import torch.nn as nn

labels = ['财经', '时政', '房产', '科技', '教育', '时尚', '游戏', '家居', '体育', '娱乐']
label2idx = {label: idx for idx, label in enumerate(labels)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(tud.Dataset):
    def __init__(self, data, max_len, model_name, with_label=False):
        super(MyDataset, self).__init__()
        self.data = data  # pandas dataframe
        # self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.with_label = with_label

    def __getitem__(self, item):
        sentences = str(self.data.loc[item, 'content'])
        encoded_sents = self.tokenizer.encode_plus(text=sentences,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=self.max_len,
                                                   return_tensors='pt')
        input_ids = encoded_sents['input_ids'].squeeze(0)
        attn_mask = encoded_sents['attention_mask'].squeeze(0)
        token_type_ids = encoded_sents['token_type_ids'].squeeze(0)
        if self.with_label:
            labels = self.data.loc[item, 'class_label']
            return input_ids, attn_mask, token_type_ids, labels
        return input_ids, attn_mask, token_type_ids

    def __len__(self):
        return len(self.data)


class MyModel(nn.Module):
    def __init__(self, model_name, output_size, hidden_size, dropout):
        super(MyModel, self).__init__()
        # self.bert = XLNetModel.from_pretrained(model_name, return_dict=True)
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(labels))
        )

    def forward(self, input_ids, attn_mask, token_type_ids):
        enc_output = self.bert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        # (batch_sz, seq_len, hidden_sz)
        # dec_input = enc_output.last_hidden_state[:,0,:]
        dec_input = enc_output.pooler_output
        dec_output = self.fc(dec_input)
        return dec_output
