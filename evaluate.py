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
from argparse import Namespace 

from ARCTIC.ARCTIC_model import ARCTIC,AttentionDecoder,AdditiveAttention,ImageEncoder
from ARCTIC.ARCTIC_dataloader import ImageTextDataset ,mktrainval,cap_to_wvec,wvec_to_cap,wvec_to_capls

max_cider=0.0043918896512309125
max_spice=0.18703616844830037
img_path = f'../data/deepfashion-multimodal/images'

def cider_d(reference_list, candidate_list, n=4):
    def count_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def compute_cider_d(reference_list, candidate_list, n):
        cider_d_scores = []
        for refs, cand in zip(reference_list, candidate_list):
            cider_d_score = 0.0
            for i in range(1, n + 1):
                cand_ngrams = count_ngrams(cand, i)
                ref_ngrams_list = [count_ngrams(ref, i) for ref in refs]

                total_ref_ngrams = [ngram for ref_ngrams in ref_ngrams_list for ngram in ref_ngrams]

                count_cand = 0
                count_clip = 0

                for ngram in cand_ngrams:
                    count_cand += 1
                    if ngram in total_ref_ngrams:
                        count_clip += 1

                precision = count_clip / count_cand if count_cand > 0 else 0.0
                recall = count_clip / len(total_ref_ngrams) if len(total_ref_ngrams) > 0 else 0.0

                beta = 1.0
                f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if precision + recall > 0 else 0.0

                cider_d_score += f_score

            cider_d_score /= n
            cider_d_scores.append(cider_d_score)

        return cider_d_scores

    reference_tokens_list = reference_list
    candidate_tokens_list = candidate_list

    scores = compute_cider_d(reference_tokens_list, candidate_tokens_list, n)

    return np.mean(scores)
def spice(reference_list, candidate_list, idf=None, beta=3):
    def tokenize(sentence):
        return sentence.lower().split()

    def count_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def compute_spice_score(reference, candidate, idf, beta):
        reference_tokens = reference
        candidate_tokens = candidate

        reference_ngrams = [count_ngrams(reference_tokens, i) for i in range(1, beta + 1)]
        candidate_ngrams = [count_ngrams(candidate_tokens, i) for i in range(1, beta + 1)]

        precision_scores = []
        recall_scores = []

        for i in range(beta):
            common_ngrams = set(candidate_ngrams[i]) & set(reference_ngrams[i])

            precision = len(common_ngrams) / len(candidate_ngrams[i]) if len(candidate_ngrams[i]) > 0 else 0.0
            recall = len(common_ngrams) / len(reference_ngrams[i]) if len(reference_ngrams[i]) > 0 else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)

        precision_avg = np.mean(precision_scores)
        recall_avg = np.mean(recall_scores)

        spice_score = (precision_avg * recall_avg) / (precision_avg + recall_avg) if precision_avg + recall_avg > 0 else 0.0

        if idf:
            spice_score *= np.exp(np.sum([idf[token] for token in common_ngrams]) / len(candidate_tokens))

        return spice_score

    if idf is None:
        idf = {}

    spice_scores = []

    for reference, candidate in zip(reference_list, candidate_list):
        spice_score = compute_spice_score(reference, candidate, idf, beta)
        spice_scores.append(spice_score)

    return np.mean(spice_scores)
def get_BLEU_score(cands, refs): #获取BLEU分数
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//1)*1 : (idx//1)*1+1])#每个候选文本对应cpi==1条参考文本
    bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
    return bleu4
def get_CIDER_D_score(vocab,cands, refs): #获得CIDER-D分数
    refs_ = [wvec_to_capls(vocab,ref) for ref in refs]
    cands_ = [wvec_to_capls(vocab,cand) for cand in cands]
    return cider_d(refs_, cands_)
def get_SPICE_score(vocab,cands, refs): #获得SPICE分数
    refs_ = [wvec_to_cap(vocab,ref) for ref in refs]
    cands_ = [wvec_to_cap(vocab,cand) for cand in cands]
    return spice(refs_, cands_)


def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

def evaluate_ARCTIC(model_path="../best_arctic.ckpt"):
    model=torch.load(model_path)["model"] #加载模型
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
    config=ARCTIC_config
    img_path = f'../data/deepfashion-multimodal/images'
    _ ,test_loader=mktrainval(data_dir='../data/deepfashion-multimodal',\
                                            vocab_path='../data/deepfashion-multimodal/vocab.json',\
                                            batch_size=3,workers=0) 
    data_loader=test_loader
    model.eval()
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
    
    bleu4_score = get_BLEU_score(cands, refs)
    cider_d_score = get_CIDER_D_score(model.vocab,cands, refs)
    spice_score= get_SPICE_score(model.vocab,cands, refs)
    print(f"@@@实际值 BLEU:{bleu4_score}|CIDEr-D:{cider_d_score}|SPICE:{spice_score}")
    print(f"@@@相对值（0-1） BLEU:{bleu4_score}|CIDEr-D:{cider_d_score/max_cider}|SPICE:{spice_score/max_spice}")

evaluate_ARCTIC()