import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
import torchvision.transforms as transforms
import copy
import pickle
import numpy as np
import time
import sys

from config import Config
from utils_func import *
from utils_data import DLoader
from model_vgg import *

datanum = 40000

class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.enc_lr = self.config.enc_lr
        self.dec_lr = self.config.dec_lr

        # model params
        self.img_size = self.config.img_size
        self.max_len = self.config.max_len

        # for reproducibility
        torch.manual_seed(999)

        # set transforms (ImageNet mean, std because pre-trained ResNet101 trained by ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.trans = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        # make dataset
        self.img_folder = self.base_path + 'data/images/'
        self.caption_file = self.base_path + 'data/captions.txt'
        self.all_pairs = collect_all_pairs(self.caption_file)

        self.all_pairs = self.all_pairs[:datanum]

        self.trainset_id, self.valset_id = make_dataset_ids(len(self.all_pairs), datanum // 5)
        self.tokenizer = Tokenizer(self.config, self.all_pairs, self.trainset_id)

        if self.mode == 'train':
            self.trainset = DLoader(self.img_folder, self.all_pairs, self.trans, self.trainset_id, self.tokenizer, self.max_len)
            self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.valset = DLoader(self.img_folder, self.all_pairs, self.trans, self.valset_id, self.tokenizer, self.max_len)
        self.dataloaders['test'] = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # model, optimizer, loss
        self.encoder = Encoder(self.config).to(self.device)
        self.decoder = Decoder(self.config, self.tokenizer).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if self.mode == 'train':
            self.enc_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=self.enc_lr)
            self.dec_optimizer = optim.Adam(self.decoder.parameters(), lr=self.dec_lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.encoder.load_state_dict(self.check_point['model']['encoder'])
                self.decoder.load_state_dict(self.check_point['model']['decoder'])
                self.enc_optimizer.load_state_dict(self.check_point['optimizer']['encoder'])
                self.dec_optimizer.load_state_dict(self.check_point['optimizer']['decoder'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test' or self.mode == 'inference':
            self.trans4attn = transforms.Compose([
                transforms.Resize((252, 252)),
                transforms.ToTensor()])
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.encoder.load_state_dict(self.check_point['model']['encoder'])
            self.decoder.load_state_dict(self.check_point['model']['decoder'])
            self.encoder.eval()
            self.decoder.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': [], 'topk_acc': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                else:
                    self.encoder.eval()
                    self.decoder.eval()

                total_loss, total_acc = 0, 0
                all_val_trg, all_val_output = [], []
                for i, (img, cap, _) in enumerate(self.dataloaders[phase]):
                    batch_size = img.size(0)
                    img, cap = img.to(self.device), cap.to(self.device)
                    self.enc_optimizer.zero_grad()
                    self.dec_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        enc_output, hidden = self.encoder(img)
                        
                        decoder_all_output, decoder_all_score = [], []
                        for j in range(self.max_len):
                            trg_word = cap[:, j].unsqueeze(1)
                            dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output)
                            decoder_all_output.append(dec_output)

                        decoder_all_output = torch.cat(decoder_all_output, dim=1)

                        loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), cap[:, 1:].reshape(-1))
                        if self.config.is_attn:
                            decoder_all_score = torch.cat(decoder_all_score, dim=2)
                            loss += self.config.regularization_lambda * ((1. - torch.sum(decoder_all_score, dim=2)) ** 2).mean()
                        acc = topk_accuracy(decoder_all_output[:, :-1, :], cap[:, 1:], self.config.topk, self.tokenizer.eos_token_id)
                        if phase == 'train':
                            loss.backward()
                            self.enc_optimizer.step()
                            self.dec_optimizer.step()
                        else:
                            all_val_trg.append(cap.detach().cpu())
                            all_val_output.append(decoder_all_output.detach().cpu())

                    total_loss += loss.item()*batch_size
                    total_acc += acc * batch_size
                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}, top-{} acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), self.config.topk, acc))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
                print('{} loss: {:4f}, top-{} acc: {:4f}\n'.format(phase, epoch_loss, self.config.topk, epoch_acc))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)

                    # print examples
                    print_samples(cap, decoder_all_output, self.tokenizer)

                    # calculate scores
                    all_val_trg, all_val_output = tensor2list(all_val_trg, all_val_output, self.tokenizer)
                    val_score_history['bleu2'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 2))
                    val_score_history['bleu4'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 4))
                    val_score_history['nist2'].append(cal_scores(all_val_trg, all_val_output, 'nist', 2))
                    val_score_history['nist4'].append(cal_scores(all_val_trg, all_val_output, 'nist', 4))
                    val_score_history['topk_acc'].append(epoch_acc)
                    print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(val_score_history['bleu2'][-1], val_score_history['bleu4'][-1], val_score_history['nist2'][-1], val_score_history['nist4'][-1]))
                    
                    # save best model
                    early_stop += 1
                    if best_val_bleu < val_score_history['bleu4'][-1]:
                        early_stop = 0
                        best_val_bleu = val_score_history['bleu4'][-1]
                        best_enc_wts = copy.deepcopy(self.encoder.state_dict())
                        best_dec_wts = copy.deepcopy(self.decoder.state_dict())
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, [self.encoder, self.decoder], [self.enc_optimizer, self.dec_optimizer])

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val bleu: {:4f}, best epoch: {:d}\n'.format(best_val_bleu, best_epoch))
        self.model = {'encoder': self.encoder.load_state_dict(best_enc_wts), 'decoder': self.decoder.load_state_dict(best_dec_wts)}
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'val_score_history': val_score_history}
        return self.model, self.loss_data
    

    def test(self):        
        # statistics of the test set
        phase = 'test'
        total_loss = 0
        all_val_trg, all_val_output, all_val_score = [], [], []

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()

            for img, cap, _ in self.dataloaders[phase]:
                batch = img.size(0)
                img, cap = img.to(self.device), cap.to(self.device)
                enc_output, hidden = self.encoder(img)
                
                decoder_all_output =[]
                for j in range(self.max_len):
                    trg_word = cap[:, j].unsqueeze(1)
                    dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output)
                    decoder_all_output.append(dec_output)
                    # if self.config.is_attn:
                    #     decoder_all_score.append(score)

                decoder_all_output = torch.cat(decoder_all_output, dim=1)
                # if self.config.is_attn:
                #     decoder_all_score = torch.stack(decoder_all_score, dim=0) # cat을 stack으로 변경 , dim = 2 -> 0으로 변경

                loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), cap[:, 1:].reshape(-1))
                # loss += self.config.regularization_lambda * ((1. - torch.sum(decoder_all_score, dim=2)) ** 1).mean()
                
                all_val_trg.append(cap.detach().cpu())
                all_val_output.append(decoder_all_output.detach().cpu())
                # if self.config.is_attn:
                #     all_val_score.append(decoder_all_score.detach().cpu())

                total_loss += loss.item()*batch

        # calculate loss and ppl
        total_loss = total_loss / len(self.dataloaders[phase].dataset)
        print('loss: {}, ppl: {}'.format(total_loss, np.exp(total_loss)))

        # calculate scores
        all_val_trg_l, all_val_output_l = tensor2list(all_val_trg, all_val_output, self.tokenizer)
        bleu2 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 2)
        bleu4 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 4)
        nist2 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 2)
        nist4 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 4)
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))


    def inference(self, result_num, model_name):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            sys.exit()
        
        # statistics of IMDb test set
        phase = 'test'
        total_loss = 0
        all_val_trg, all_val_output, all_val_score = [], [], []

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            all_ids = []

            for img, cap, id in self.dataloaders[phase]:
                all_ids.append(id.cpu())
                batch = img.size(0)
                img, cap = img.to(self.device), cap.to(self.device)
                enc_output, hidden = self.encoder(img)
                
                decoder_all_output, decoder_all_score = [], []
                for j in range(self.max_len):
                    if j == 0:
                        trg_word = cap[:, j].unsqueeze(1)
                        dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output)
                    else:
                        trg_word = torch.argmax(dec_output, dim=-1)
                        dec_output, hidden, score = self.decoder(trg_word.detach(), hidden, enc_output)
                    
                    decoder_all_output.append(dec_output)
                    if self.config.is_attn:
                        decoder_all_score.append(score)

                decoder_all_output = torch.cat(decoder_all_output, dim=1)
                if self.config.is_attn:
                    decoder_all_score = torch.cat(decoder_all_score, dim=2)

                loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), cap[:, 1:].reshape(-1))
                loss += self.config.regularization_lambda * ((1. - torch.sum(decoder_all_score, dim=2)) ** 2).mean()
                
                all_val_trg.append(cap.detach().cpu())
                all_val_output.append(decoder_all_output.detach().cpu())
                if self.config.is_attn:
                    all_val_score.append(decoder_all_score.detach().cpu())

                total_loss += loss.item()*batch
            all_ids = torch.cat(all_ids, dim=0).tolist()

        # calculate loss and ppl
        total_loss = total_loss / len(self.dataloaders[phase].dataset)
        print('Inference Score')
        print('loss: {}, ppl: {}'.format(total_loss, np.exp(total_loss)))

        # calculate scores
        all_val_trg_l, all_val_output_l = tensor2list(all_val_trg, all_val_output, self.tokenizer)
        bleu2 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 2)
        bleu4 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 4)
        nist2 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 2)
        nist4 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}\n\n'.format(bleu2, bleu4, nist2, nist4))

        # show results examples
        random.seed(int(1000*time.time())%(2**32))
        all_val_trg = torch.cat(all_val_trg, dim=0)
        all_val_output = torch.argmax(torch.cat(all_val_output, dim=0), dim=2)
        ids = random.sample(list(range(all_val_trg.size(0))), 10)
        img_id = [self.img_folder+self.all_pairs[j][0] for j in [all_ids[i] for i in ids]]
        gt, pred = print_samples(all_val_trg, all_val_output, self.tokenizer, result_num, ids)
        if self.config.is_attn:
            all_val_score = torch.cat(all_val_score, dim=0)
            pred_l = [len(self.tokenizer.tokenize(s)) for s in pred]
            attn_img = [all_val_score[i, :, :l] for i, l in zip(ids, pred_l)]

        # save result figures
        results_save_path = self.base_path + 'result/' + model_name + '/'
        save_figures(img_id, gt, pred, results_save_path)
        if self.config.is_attn:
            save_attn_figures(img_id, attn_img, pred, results_save_path, self.trans4attn, self.config.enc_hidden_size)