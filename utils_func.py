import torch
import torch.nn.functional as F
import re
import random
import matplotlib.pyplot as plt
import math
import random
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
import pandas as pd
import os
from PIL import Image
import skimage.transform



def save_checkpoint(file, model, optimizer):
    state = {'model': {'encoder':model[0].state_dict(), 'decoder': model[1].state_dict()}, 'optimizer': {'encoder':optimizer[0].state_dict(), 'decoder': optimizer[1].state_dict()}}
    torch.save(state, file)
    print('model pt file is being saved\n')


def print_samples(trg, output, tokenizer, show_n=3, idx=None):
    all_t, all_o = [], []
    trg, output = trg.detach().cpu(), output.detach().cpu()
    if idx == None:
        idx = random.sample(list(range(trg.size(0))), show_n)
        print('-'*50)
        for i in idx:
            t, o = trg[i, 1:].tolist(), torch.argmax(output[i, :-1], dim=1).tolist()
            t, o = tokenizer.decode(t), tokenizer.decode(o)
            t, o = ' '.join(t.split()), ' '.join(o.split())
            print('gt  : {}'.format(t))
            print('pred: {}\n'.format(o))
            all_t.append(t); all_o.append(o)
        print('-'*50 + '\n')
    else:
        print('-'*50)
        for i in idx:
            t, o = trg[i, 1:].tolist(), output[i, :-1].tolist()
            t, o = tokenizer.decode(t), tokenizer.decode(o)
            t, o = ' '.join(t.split()), ' '.join(o.split())
            print('gt  : {}'.format(t))
            print('pred: {}\n'.format(o))
            all_t.append(t); all_o.append(o)
        print('-'*50 + '\n')
    return all_t, all_o



def bleu_score(ref, pred, weights):
    return corpus_bleu(ref, pred, weights)


def nist_score(ref, pred, n):
    return corpus_nist(ref, pred, n)


def cal_scores(ref, pred, type, n_gram):
    assert type in ['bleu', 'nist']
    if type == 'bleu':
        wts = tuple([1/n_gram]*n_gram)
        return bleu_score(ref, pred, wts)
    return nist_score(ref, pred, n_gram)



def tensor2list(ref, pred, tokenizer):
    ref, pred = torch.cat(ref, dim=0)[:, 1:], torch.cat(pred, dim=0)[:, :-1]
    ref = [[tokenizer.decode(ref[i].tolist()).split()] for i in range(ref.size(0))]
    pred = [tokenizer.decode(torch.argmax(pred[i], dim=1).tolist()).split() for i in range(pred.size(0))]
    return ref, pred


def collect_all_pairs(caption_file):
    captions = pd.read_csv(caption_file)
    all_pairs = [[img, preprocessing(cap.lower())] for img, cap in zip(captions['image'], captions['caption'])]
    return all_pairs
    

def make_dataset_ids(total_l, valset_l):
    random.seed(999)
    all_id = list(range(total_l))
    trainset_id = random.sample(all_id, total_l - valset_l)
    valset_id = list(set(all_id) - set(trainset_id))
    return trainset_id, valset_id


def preprocessing(s):
    s = re.sub('[#$%&()*+\-/:;<=>@\[\]^_`{|}~\'".,?!]', '', s).lower()
    s = ' '.join(s.split())
    return s


def topk_accuracy(pred, target, k, eos_token_id):
    pred, target = pred.detach().cpu(), target.detach().cpu()

    total_correct = 0
    batch_size = 0
    for i in range(target.size(0)):
        l = target[i].tolist().index(eos_token_id)
        _, idx = torch.topk(pred[i, :l], k, dim=1)
        correct = idx.eq(target[i, :l].unsqueeze(1).expand_as(idx))
        total_correct += correct.view(-1).float().sum()
        batch_size += l
    return total_correct.item() * (100.0 / batch_size)


def save_figures(img_id, gt, pred, save_path):
    os.makedirs(save_path, exist_ok=True)

    for i, (img, g, p) in enumerate(zip(img_id, gt, pred)):
        plt.figure()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        img = plt.imread(img)
        h, w, _ = img.shape

        plt.imshow(img)
        plt.text(w/2, h+20, 'gt: '+g, horizontalalignment='center')
        plt.text(w/2, h+40, 'pred: '+p, horizontalalignment='center')
        plt.savefig(save_path + 'result_' + str(i+1) + '.jpg')


def save_attn_figures(img_id, attn_img, pred, save_path, trans, enc_hidden_size):
    os.makedirs(save_path, exist_ok=True)
    mybox={'facecolor':'w','boxstyle':'square','alpha':1}

    for i, (img, attn, p) in enumerate(zip(img_id, attn_img, pred)):
        img = trans(Image.open(img)).permute(1, 2, 0)
        attn = attn.view(enc_hidden_size, enc_hidden_size, -1)
        p = p.split()

        plt.figure(figsize=(4*3, math.ceil(len(p)/4)*3))
        for j in range(attn.size(-1)):
            score = skimage.transform.pyramid_expand(attn[:, :, j].numpy(), upscale=18, sigma=8)
            plt.subplot(math.ceil(len(p)/4), 4, j+1)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(img)
            plt.imshow(score, cmap='gray', alpha=0.7)
            plt.text(5, 15, p[j], bbox=mybox)
        plt.savefig(save_path + 'result_attn_' + str(i+1) + '.jpg')
