import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import easydict
from datapreprocess import PoemDataSet
from model import MyPoetryModel_tanh
from myutils import accuracy,AvgrageMeter
import matplotlib.pyplot as plt

f = open("./configs.yaml", 'r')
configs = yaml.safe_load(f)
Config = easydict.EasyDict(configs)
f.close()
poem_ds = PoemDataSet(Config.data_path, 48)
ix2word = poem_ds.ix2word
word2ix = poem_ds.word2ix
print(poem_ds[0])

poem_loader =  DataLoader(poem_ds,
                     batch_size=16,
                     shuffle=True,
                     num_workers=0)





def train( epochs, train_loader, device, model, criterion, optimizer,scheduler,tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    train_loss_list = []
    train_accuracy_list = []
    for epoch in range(epochs):
        train_loss = 0.0
        train_loader = train_loader
        #train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
#             print(' '.join(ix2word[inputs.view(-1)[k] for k in inputs.view(-1).shape.item()]))
            labels = labels.view(-1) # 因为outputs经过平整，所以labels也要平整来对齐
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs,hidden = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            _,pred = outputs.topk(1)
#             print(get_word(pred))
#             print(get_word(labels))
            prec1, prec2= accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'epoch':epoch,'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            #print(postfix)


            # ternsorboard 曲线绘制
            # if os.path.exists(Config.tensorboard_path) == False:
            #     os.mkdir(Config.tensorboard_path)
            # writer = SummaryWriter(tensorboard_path)
            # writer.add_scalar('Train/Loss', loss.item(), epoch)
            # writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            # writer.flush()
        train_loss_list.append(loss.item())
        train_accuracy_list.append(top1.avg)
            # if os.path.exists(Config.model_save_path) == False:
            #     os.mkdir(Config.model_save_path)
            # torch.save(model.state_dict(), Config.model_save_path+"/tmppoem.pth")
        scheduler.step()
        print(postfix)
        # 模型保存
        if os.path.exists(Config.model_save_path) == False:
            os.mkdir(Config.model_save_path)
        torch.save(model.state_dict(), Config.model_save_path+"/tmppoem.pth")

        print('Finished Training')
    x1 = range(0, Config.epoch)
    y1 = train_loss_list
    y3 = train_accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')


    plt.legend(["train_loss"])
    plt.title('Loss vs. epoches')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y3, '.-')
    plt.legend(["train_accuracy"])
    plt.xlabel('Accuracy vs. epoches')
    plt.ylabel('Accuracy')
    plt.show()

    plt.savefig("poem" + "_accuracy_loss.jpg")

model = MyPoetryModel_tanh(len(word2ix),
                  embedding_dim=Config.embedding_dim,
                  hidden_dim=Config.hidden_dim)
# if os.path.exists()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = Config.epoch
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Config.step_size,gamma=Config.lr_gamma)#学习率调整
criterion = nn.CrossEntropyLoss()

import shutil
if os.path.exists(Config.tensorboard_path):
    shutil.rmtree(Config.tensorboard_path,ignore_errors=True)
    os.mkdir(Config.tensorboard_path)
if os.path.exists(Config.model_save_path+"/tmppoem.pth") == True:
    model.load_state_dict(torch.load(Config.model_save_path+"/tmppoem.pth"))

train(epochs, poem_loader, device, model, criterion, optimizer,scheduler, Config.tensorboard_path)
#模型保存
if os.path.exists(Config.model_save_path) == False:
    os.mkdir(Config.model_save_path)
torch.save(model.state_dict(), Config.model_save_path)

model.load_state_dict(torch.load(Config.model_save_path))

def generate(model, start_words, ix2word, word2ix,device):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()

    #最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers*1,1,Config.hidden_dim),dtype=torch.float)
    input = input.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
            for i in range(48):#诗的长度
                output, hidden = model(input, hidden)
                # 如果在给定的句首中，input为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
               # 否则将output作为下一个input进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()  # 输出的预测的字
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)
                if w == '<EOP>':  # 输出了结束标志就退出
                    del results[-1]
                    break
    return results



results = generate(model,'雨', ix2word,word2ix,device)
print(' '.join(i for i in results))
results = generate(model,'湖光秋月两相得', ix2word,word2ix,device)
print(' '.join(i for i in results))
results = generate(model,'人生得意须尽欢，', ix2word,word2ix,device)
print(' '.join(i for i in results))
results = generate(model,'万里悲秋常作客，', ix2word,word2ix,device)
print(' '.join(i for i in results))
results = generate(model,'风急天高猿啸哀，渚清沙白鸟飞回。', ix2word,word2ix,device)
print(' '.join(i for i in results))
results = generate(model,'千山鸟飞绝，万径人踪灭。', ix2word,word2ix,device)
print(' '.join(i for i in results))
