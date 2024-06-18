import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.enc_hidden_size = config.enc_hidden_size
        self.dec_hidden_size = config.dec_hidden_size
        self.dec_num_layers = config.dec_num_layers
        self.pixel_size = self.enc_hidden_size * self.enc_hidden_size

        base_model = resnet101(pretrained=True, progress=False)
        base_model = list(base_model.children())[:-2]
        self.resnet = nn.Sequential(*base_model)  # output size: B x 2048 x H/32 x W/32
        self.pooling = nn.AdaptiveAvgPool2d((self.enc_hidden_size, self.enc_hidden_size))

        self.relu = nn.ReLU()
        self.hidden_dim_changer = nn.Sequential(
            nn.Linear(self.pixel_size, self.dec_num_layers),
            nn.ReLU()
        )
        self.h_mlp = nn.Linear(2048, self.dec_hidden_size)
        self.c_mlp = nn.Linear(2048, self.dec_hidden_size)

        self.fine_tune(True)


    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


    def forward(self, x):
        batch_size = x.size(0)

        x = self.resnet(x)
        x = self.pooling(x)
        x = x.view(batch_size, 2048, -1)

        if self.dec_num_layers != 1:
            tmp = self.hidden_dim_changer(self.relu(x))
        else:
            tmp = torch.mean(x, dim=2, keepdim=True)
        tmp = torch.permute(tmp, (2, 0, 1))
        h0 = self.h_mlp(tmp)
        c0 = self.c_mlp(tmp)
        return x, (h0, c0)



class Decoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(Decoder, self).__init__()
        self.pixel_size = config.enc_hidden_size * config.enc_hidden_size
        self.dec_hidden_size = config.dec_hidden_size
        self.dec_num_layers = config.dec_num_layers
        self.dropout = config.dropout
        self.is_attn = config.is_attn
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        if self.is_attn:
            self.attention = Attention(self.dec_hidden_size)
        self.input_size = self.dec_hidden_size + 2048 if self.is_attn else self.dec_hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.dec_hidden_size,
                            num_layers=self.dec_num_layers,
                            batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout) 
        self.relu = nn.ReLU()
        self.beta = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dec_hidden_size, 2048),
            nn.Sigmoid()
        )     
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dec_hidden_size, self.vocab_size)
        )

        self.embedding.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.bias.data.fill_(0)
            m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m, nn.Embedding):
            m.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x, hidden, enc_output):
        x = self.embedding(x)
        score = None

        gate = self.beta(hidden[0][-1])
        if self.is_attn:
            enc_output, score = self.attention(self.relu(enc_output), self.relu(hidden[0][-1]))
            enc_output = gate*enc_output
            x = torch.cat((x, enc_output.unsqueeze(1)), dim=-1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden, score



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.enc_wts = nn.Sequential(
            nn.Linear(2048, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.dec_wts = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.score_wts = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, enc_output, dec_hidden):
        enc_output = torch.permute(enc_output, (0, 2, 1))
        
        score = self.tanh(self.enc_wts(enc_output) + self.dec_wts(dec_hidden).unsqueeze(1))
        score = self.score_wts(score)
        score = F.softmax(score, dim=1)

        enc_output = torch.permute(enc_output, (0, 2, 1))
        enc_output = torch.bmm(enc_output, score).squeeze(-1)
        return enc_output, score