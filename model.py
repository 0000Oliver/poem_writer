import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml
import easydict
f = open("./configs.yaml", 'r')
configs = yaml.safe_load(f)
Config = easydict.EasyDict(configs)
class MyPoetryModel_tanh(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel_tanh, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True,dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,vocab_size)
#         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, hidden=None):
            embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
            batch_size, seq_len = input.size()
            if hidden is None:
                h_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden
            output, hidden = self.lstm(embeds, (h_0, c_0))#hidden 是h,和c 这两个隐状态
            output = torch.tanh(self.fc1(output))
            output = torch.tanh(self.fc2(output))
            output = self.fc3(output)
            output = output.reshape(batch_size * seq_len, -1)
            return output,hidden
f.close()