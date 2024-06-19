from collections import Counter
from tqdm import tqdm


class Tokenizer:
    def __init__(self, config, all_pairs, trainset_id):
        self.all_pairs, self.trainset_id = all_pairs, trainset_id
        self.vocab_size = config.vocab_size
        self.pad_token, self.sos_token, self.eos_token, self.unk_token = '[PAD]', '[SOS]', '[EOS]', '[UNK]'
        self.pad_token_id, self.sos_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self.word2idx = {self.pad_token: self.pad_token_id, self.sos_token: self.sos_token_id, self.eos_token: self.eos_token_id, self.unk_token: self.unk_token_id}
        self.idx2word = {self.pad_token_id: self.pad_token, self.sos_token_id: self.sos_token, self.eos_token_id: self.eos_token, self.unk_token_id: self.unk_token}

        # count the word frequency
        self.train_caption = [all_pairs[id][1] for id in self.trainset_id]
        self.word_freq = Counter()
        for cap in self.train_caption:
            self.word_freq.update(cap.split())

        # update vocab
        for word, _ in tqdm(self.word_freq.most_common(self.vocab_size-len(self.word2idx)), desc='make tokenizer'):
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

        assert len(self.word2idx) == len(self.idx2word)
        self.vocab_size = min(self.vocab_size, len(self.word2idx))


    def tokenize(self, s):
        return s.split()


    def encode(self, s):
        s = [self.word2idx[w] if w in self.word2idx else self.word2idx[self.unk_token] for w in self.tokenize(s)]
        return s


    def decode(self, tok):
        s = [self.idx2word[t] for t in tok]
        try:
            s = ' '.join(s[:tok.index(self.eos_token_id)])
        except ValueError:
            s = ' '.join(s)
        return s