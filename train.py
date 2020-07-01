from model import PoetryModel
from torch import optim
from torch import nn
#import tqdm
import torch
import torchnet.meter as meter
from torch.autograd import Variable
import  os
import ipdb
import numpy as np
from torch.utils.data import DataLoader,Dataset

import yaml
import easydict as Edict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import re

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
        datas = np.load(self.poem_path)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix

## topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True) #使用topk来获得前k个的索引
    pred = pred.t() # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred)) # 与正确标签序列形成的矩阵相比，生成True/False矩阵
#     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn

def generate(model,start_words,ix2word,word2ix,Config):
    result = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1,1).long()
    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(Config.max_gen_len):
            output,hidden = model(input,hidden)
            # 如果在给定的句首中，input为句首中的下一个字
            if i<start_words_len:
                w = result[i]
                input= input.data.new([word2ix[w]]).view(1,1)
            # 否则将output作为下一个input进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                result.append(w)
                input = input.data.new([top_index]).view(1,1)
            if w =='<EOP>':
                del result[-1]
                break
        return result
def train(Config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas = np.load("data/chinese-poetry-master/tang .npz",allow_pickle=True)


    data = datas["data"]
    ix2word =datas['ix2word'].item()
    word2ix =datas['word2ix'].item()
    data = torch.from_numpy(data)
    print(data.shape)
    # #去掉空格
    # t_data = data.view(-1)
    # flat_data = t_data.numpy()
    # no_space_data = []
    # for i in flat_data:
    #     if (i != 8292):
    #         no_space_data.append(i)
    # slice_size = 48
    # txt = [no_space_data[i:i+slice_size] for i in range(0,len(no_space_data),slice_size)]
    # txt = np.array(txt[:-1])#去掉最后一个不够48的数据
    # txt = torch.from_numpy(txt).long()
    # print(txt.shape)
    #datas = PoemDataSet(Config.data_path, 48)
    #data = datas.no_space_data#datas['data']
    #ix2word = datas.ix2word#datas['ix2word'].item()
    #word2ix = datas.word2ix#datas['word2ix'].item()

    dataLoader = DataLoader(data, batch_size=Config.batch_size, shuffle=Config.shuffle, num_workers=Config.num_workers)

    model = PoetryModel(len(word2ix),
                        embedding_dim=Config.embedding_dim,
                        hidden_dim=  Config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(),lr = Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
    crierion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()
    top1 = meter.AverageValueMeter()
    #top1 = utils.AverageMeter()
    # if Config.model_path:
    #     model.load_state_dict(torch.load(Config.model_path))
    train_loss_list = []
    train_accuracy_list = []
    for epoch in range(Config.epoch):

        loss_meter.reset()
        top1.reset()
        for ii,data_ in enumerate(dataLoader):#tqdm.tqdm(enumerate(dataLoader)):
            #inputs, labels =Variable(data_[0]), Variable(data_[1])#.to(device)
            data_ = data_.long().transpose(1,0).contiguous()
            inputs,labels = Variable(data_[:-1,:]),Variable(data_[1:,:])
            print(inputs.size(1))
            optimizer.zero_grad()

            # 上边一句，将输入的诗句错开一个字，形成训练和目标
            output,_ = model(inputs)
            loss = crierion(output, labels.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            _, pred = output.topk(1)
            prec1, prec2 = accuracy(output, labels, topk=(1, 2))
            n = inputs.size(0)
            top1.add(prec1.item())
            #data = data.long().transpose(1,0).contiguous()

            if(1+ii)%Config.plot_every ==0:

                if os.path.exists(Config.debug_file):
                    ipdb.set_trace()


                # 下面是对目前模型情况的测试，诗歌原文、
                #print(inputs.size(1))
                #print(inputs.numpy()[:1].shape)
                # poetrys = [[ix2word[_word] for _word in inputs.numpy()[:, _iii]]
                #            for _iii in range(inputs.size(1))][0]
                # poetrys =["".join(poetry) for poetry in poetrys]
                # print("origen")
                #print(poetrys)
                # 上面句子嵌套了两个循环，主要是将诗歌索引的前十六个字变成原文

                gen_poetries = []
                start= u"春江花月夜凉如水"
                gen_poetry = "".join(generate(model, start, ix2word, word2ix, Config))
                # for word in list(u"春江花月夜凉如水"):
                #     gen_poetry = "".join(generate(model, word, ix2word, word2ix,Config))
                #     gen_poetries.append(gen_poetry)
                # gen_poetries="</br>".join(["".join(poetry) for poetry in gen_poetries])
                print("genetate")
                print(gen_poetry)

            # if os.path.exists(Config.tensorboard_path) == False:
            #     os.mkdir(Config.tensorboard_path)
            # writer = SummaryWriter(Config.tensorboard_path)
            # writer.add_scalar('Train/Loss', loss.item(), epoch)
            # writer.add_scalar('Train/Accuracy', 100*prec1.item()/output.size(0), epoch)
            #
            # writer.flush()
        train_loss_list.append(loss.item())
        train_accuracy_list.append(100 * prec1.item() / output.size(0))
        print('train %d epoch loss: %.3f acc: %.3f ' % (epoch + 1, loss_meter.mean, 100*top1.mean/output.size(0)))
        scheduler.step()
    x1 = range(0, configs.epoch)
    y1 = train_loss_list

    y3 = train_accuracy_list

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')

    plt.legend(["train_loss"])
    plt.title('Loss vs. epoches')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y3, '.-')
    plt.legend("train_accuracy")
    plt.xlabel('Accuracy vs. epoches')
    plt.ylabel('Accuracy')
    plt.show()

    plt.savefig("pw_LSTM" + "_accuracy_loss.jpg")
    torch.save(model.state_dict(), "%s_%s.pth" % (Config.model_prefix, epoch))
if __name__ =="__main__":

        with open('./configs.yaml') as f:
            configs = yaml.safe_load(f)
            configs = Edict.EasyDict(configs)
            print(configs)
            train(configs)
            import shutil

            if os.path.exists(configs.tensorboard_path):
                shutil.rmtree(configs.tensorboard_path)
                os.mkdir(configs.tensorboard_path)



