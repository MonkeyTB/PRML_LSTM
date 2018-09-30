#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.autograd as autograd		#Torch中自动计算梯度模块
import torch.nn as nn					#神经网络模块
import torch.nn.functional as F			#神经网路模块中的常用功能
import torch.optim as optim				#模型优化器模块
torch.manual_seed(1)
def prepare_char_sequence(word, to_ix):
    idxs = [to_ix[char] for char in word]
    return(torch.tensor(idxs, dtype=torch.long))


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    char_idxs = [prepare_char_sequence(w, char_to_ix) for w in seq]
    return torch.tensor(idxs, dtype=torch.long), char_idxs


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
print(word_to_ix)
print(char_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
WORD_EMBEDDING_DIM = 5
CHAR_EMBEDDING_DIM = 5
CHAR_HIDDEN_DIM = 3
WORD_HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    def __init__(self, char_embedding_dim, word_embedding_dim, char_hidden_dim, word_hidden_dim, char_size, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.word_lstm = nn.LSTM((word_embedding_dim + char_hidden_dim), word_hidden_dim)

        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.char_hidden = self.init_hidden(self.char_hidden_dim)
        self.word_hidden = self.init_hidden(self.word_hidden_dim)

    def init_hidden(self, hidden_dim):
        return(torch.zeros(1, 1, hidden_dim),
               torch.zeros(1, 1, hidden_dim))

    def forward(self, sentence):
        char_lstm_result = []
        for word in sentence[1]:
            self.char_hidden = self.init_hidden(self.char_hidden_dim)
            char_embeds = self.char_embeddings(word)
            lstm_char_out, self.char_hidden = self.char_lstm(char_embeds.view(len(word), 1, -1), self.char_hidden)
            char_lstm_result.append(lstm_char_out[-1])

        word_embeds = self.word_embeddings(sentence[0])
        char_lstm_result = torch.stack(char_lstm_result)
        lstm_in = torch.cat((word_embeds.view(len(sentence[0]), 1, -1), char_lstm_result), 2)
        lstm_out, self.hidden = self.word_lstm(lstm_in, self.word_hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
model = LSTMTagger(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, CHAR_HIDDEN_DIM, WORD_HIDDEN_DIM, len(char_to_ix), len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden(WORD_EMBEDDING_DIM)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_char_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)