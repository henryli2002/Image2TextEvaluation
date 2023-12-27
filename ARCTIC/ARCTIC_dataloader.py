from argparse import Namespace 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence # 压紧填充序列
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet101_Weights
from nltk.translate.bleu_score import corpus_bleu # BLEU评价指标
import numpy as np
import json
from torch.utils.data import Dataset
import os
from PIL import Image
from collections import Counter,defaultdict
class ImageTextDataset(Dataset):
    def __init__(self, dataset_path, vocab_path, split, captions_per_image=1, max_len=93, transform=None):

        self.split = split
        assert self.split in {'train', 'test'}
        self.cpi = captions_per_image
        self.max_len = max_len

        # 载入数据集
        with open(dataset_path, 'r') as f:
            self.data = json.load(f) #key是图片名字 value是描述
            self.data_img=list(self.data.keys())
        # 载入词典
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # PyTorch图像预处理流程
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.data_img)

    def __getitem__(self, i):
        # 第i个文本描述对应第(i // captions_per_image)张图片
        print(self.data_img[i])
        img = Image.open(img_path+"/"+self.data_img[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        c_vec=cap_to_wvec(self.vocab,self.data[self.data_img[i]])
        #加入起始和结束标志
        c_vec = [self.vocab['<start>']] + c_vec + [self.vocab['<end>']]
        caplen = len(c_vec)
        caption = torch.LongTensor(c_vec+ [self.vocab['<pad>']] * (self.max_len + 2 - caplen))
        
        return img, caption, caplen
        
    def __len__(self):
        return self.dataset_size
def mktrainval(data_dir, vocab_path, batch_size, workers=4,is_transform=True):
    train_tx = transforms.Compose([
        transforms.Resize(256), # 重置图像分辨率
        transforms.RandomCrop(224), # 随机裁剪
        transforms.ToTensor(), # 转换成Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化--三个参数为三个通道的均值和标准差
    ])
    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    no_trans=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if is_transform:
        train_set = ImageTextDataset(os.path.join(data_dir, 'train_captions.json'), vocab_path, 'train',  transform=train_tx)
        test_set = ImageTextDataset(os.path.join(data_dir, 'test_captions.json'), vocab_path, 'test', transform=val_tx)
    else:
        train_set = ImageTextDataset(os.path.join(data_dir, 'train_captions.json'), vocab_path, 'train', transform=no_trans)
        test_set = ImageTextDataset(os.path.join(data_dir, 'test_captions.json'), vocab_path, 'test', transform=no_trans)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader    
img_path = f'../data/deepfashion-multimodal/images'
def cap_to_wvec(vocab,cap):#将文本描述转换成向量
    cap.replace(",","")
    cap.replace(".","")
    cap=cap.split()
    res=[]
    for word in cap:
        if word in vocab.keys():
            res.append(vocab[word])
        else: #不在字典的词
            res.append(vocab['<unk>'])
    return res
def wvec_to_cap(vocab,wvec):#将向量转换成文本描述
    res=[]
    for word in wvec:
        for key,value in vocab.items():
            if value==word and key not in ['<start>','<end>','<pad>','<unk>']:
                res.append(key)
    res=" ".join(res)
    return res
def wvec_to_capls(vocab,wvec):#将向量转换成文本描述
    res=[]
    for word in wvec:
        for key,value in vocab.items():
            if value==word and key not in ['<start>','<end>','<pad>','<unk>']:
                res.append(key)
    return res