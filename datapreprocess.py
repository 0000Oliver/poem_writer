
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import os
def prepareData():
    datas = np.load("tang .npz")
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataLoader = DataLoader(data,batch_size=16,shuffle=True,num_workers=0)
    return dataLoader,ix2word,word2ix
class PoemDataSet(Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()

    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)

    def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ):
                no_space_data.append(i)
        return no_space_data

    def get_raw_data(self):
        #         datas = np.load(self.poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
        datas = np.load(self.poem_path,allow_pickle=True)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix
if __name__ =="__main__":

    datas = np.load("data/chinese-poetry-master/tang .npz")
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    word_data = np.zeros((1, data.shape[1]), dtype=np.str)  # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = np.random.randint(data.shape[0])
    for col in range(data.shape[1]):
        word_data[0, col] = ix2word[data[row, col]]
    print(data.shape)  # (57580, 125)
    print(word_data)  # 随机查看


