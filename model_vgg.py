import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet101

vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

def get_vgg_layer(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        
        if c == 'M': # c가 M이면 MaxPooling
            layers += [nn.MaxPool2d(kernel_size= 2)]

        else: # c가 int면 Convolution
            conv2d = nn.Conv2d(in_channels, c, kernel_size= 3, padding= 1)

            # batch normalization 적용
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace= True)]
            else:
                layers += [conv2d, nn.ReLU(inplace= True)]

            in_channels = c # 다음 layer의 in_channels로 사용

    return layers

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.enc_hidden_size = config.enc_hidden_size
        self.dec_hidden_size = config.dec_hidden_size
        self.dec_num_layers = config.dec_num_layers
        self.pixel_size = self.enc_hidden_size * self.enc_hidden_size

        # base_model = resnet101(pretrained=True, progress=False)
        # base_model = list(base_model.children())[:-2]
        base_model = get_vgg_layer(vgg_config, True)
        self.vgg = nn.Sequential(*base_model)  # output size: B x 512 x H/32 x W/32
        self.pooling = nn.AdaptiveAvgPool2d((self.enc_hidden_size, self.enc_hidden_size))

        self.relu = nn.ReLU()
        self.hidden_dim_changer = nn.Sequential(
            nn.Linear(self.pixel_size, self.dec_num_layers),
            nn.ReLU()
        )
        self.h_mlp = nn.Linear(512, self.dec_hidden_size)
        self.c_mlp = nn.Linear(512, self.dec_hidden_size)

        self.fine_tune(True)


    def fine_tune(self, fine_tune=True):
        for p in self.vgg.parameters():
            p.requires_grad = False

        for c in list(self.vgg.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


    def forward(self, x):
        batch_size = x.size(0)

        x = self.vgg(x)
        x = self.pooling(x)
        x = x.view(batch_size, 512, -1)

        if self.dec_num_layers != 1:
            tmp = self.hidden_dim_changer(self.relu(x))
        else:
            tmp = torch.mean(x, dim= 2, keepdim=True)
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
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        
        self.input_size = self.dec_hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=self.pad_token_id)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.dec_hidden_size,
                            num_layers=self.dec_num_layers,
                            batch_first=True)
        
        self.dropout_layer = nn.Dropout(self.dropout)

        self.relu = nn.ReLU()

        #self.beta = nn.Sequential(
        #    nn.ReLU(),
        #    nn.Linear(self.dec_hidden_size, 512),
        #    nn.Sigmoid()
        #)     

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

        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)

        return x, hidden, score