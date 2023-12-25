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
from ARCTIC_model import ImageEncoder ,AdditiveAttention,AttentionDecoder,ARCTIC
from ARCTIC_dataset import ImageTextDataset ,mktrainval,cap_to_wvec,wvec_to_cap,wvec_to_capls
ARCTIC_config = Namespace(
        max_len = 93,
        captions_per_image = 1,
        batch_size = 32,
        image_code_dim = 2048,
        word_dim = 512,
        hidden_size = 512,
        attention_dim = 512,
        num_layers = 1,
        encoder_learning_rate = 0.0001,
        decoder_learning_rate = 0.0005,
        num_epochs = 10,
        grad_clip = 5.0,
        alpha_weight = 1.0,
        evaluate_step = 900, # 每隔多少步在验证集上测试一次
        checkpoint = None, # 如果不为None，则利用该变量路径的模型继续训练
        best_checkpoint = 'model/ARCTIC/best_ARCTIC.ckpt', # 验证集上表现最优的模型的路径
        last_checkpoint = 'model/ARCTIC/last_ARCTIC.ckpt', # 训练完成时的模型的路径
        beam_k = 5 #束搜索的束宽
    )

class PackedCrossEntropyLoss(nn.Module):  #损失函数
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths): #压紧填充序列

        predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0] 
        targets = pack_padded_sequence(targets, lengths, batch_first=True)[0] 
        return self.loss_fn(predictions, targets) #计算损失    
def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]
def evaluate(data_loader, model, config):
    model.eval()
    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})
    cpi = config.captions_per_image
    device = next(model.parameters()).device
    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with torch.no_grad():
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), config.beam_k, config.max_len+2)
            # 候选文本
            cands.extend([filter_useless_words(text, filterd_words) for text in texts])
            # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
    # 实际上，每个候选文本对应cpi条参考文本
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
    # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
    # 即计算1-gram到4-gram的BLEU几何平均值
    bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
    model.train()
    return bleu4

if __name__ == '__main__':
# 设置GPU信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config = ARCTIC_config
    # 数据
    data_dir = 'data/deepfashion-multimodal/'
    vocab_path = 'data/deepfashion-multimodal/vocab.json'

    train_loader,test_loader=mktrainval(data_dir='data/deepfashion-multimodal',\
                                        vocab_path='data/deepfashion-multimodal/vocab.json',\
                                        batch_size=2,workers=2)

    # 模型
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # 随机初始化 或 载入已训练的模型
    start_epoch = 0
    checkpoint = config.checkpoint
    if checkpoint is None:
        model = ARCTIC(config.image_code_dim, vocab, config.word_dim, config.attention_dim, config.hidden_size, config.num_layers)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

    # 优化器
    optimizer= torch.optim.Adam(lr=0.0001, params=model.parameters())
    # 将模型拷贝至GPU，并开启训练模式
    model.to(device)
    model.train()
    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(device)
    best_res = 0
    print("开始训练")

    for epoch in range(start_epoch, config.num_epochs):
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            optimizer.zero_grad()
            # 1. 读取数据至GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            predictions, alphas, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
            loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # 4. 更新参数
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('epoch %d, step %d: loss=%.2f' % (epoch, i+1, loss.cpu()))

            state = {
                    'epoch': epoch,
                    'step': i,
                    'model': model,
                    'optimizer': optimizer
                    }
            if (i+1) % config.evaluate_step == 0:
                bleu_score = evaluate(test_loader, model, config) #在验证集上测试
                # 5. 选择模型
                if best_res < bleu_score:
                    best_res = bleu_score
                    torch.save(state, config.best_checkpoint)
                torch.save(state, config.last_checkpoint)
                print('Validation@epoch, %d, step, %d, BLEU-4=%.2f' % (epoch, i+1, bleu_score))
    checkpoint = torch.load(config.best_checkpoint)
    model = checkpoint['model']
    bleu_score = evaluate(test_loader, model, config)
    print("Evaluate on the test set with the model that has the best performance on the validation set")
    print('Epoch: %d, BLEU-4=%.2f' % (checkpoint['epoch'], bleu_score))