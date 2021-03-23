
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class BILSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, drp, n_layers):
        super(BILSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.hidden2tag = nn.Linear(2*hidden_dim, target_size)
        self.drp = drp

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        
        
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, drp, n_layers):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,n_layers, batch_first=True, dropout=drp)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.drp = drp

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        
        
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size, drp, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drp)
        self.fc = nn.Linear(hidden_dim, target_size)
        self.relu = nn.ReLU()
        self.drp = drp
        
    def forward(self, sentence):
        gru_out, _ = self.gru(sentence)
        gru_out = self.relu(gru_out)
        tag_space = self.fc(gru_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        
        

class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, drp, n_layers):
        super(RNNTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.drp = drp

    def forward(self, sentence):
        rnn_out, _ = self.rnn(sentence)
        rnn_out = self.relu(rnn_out)
        rnn_out = self.dropout(rnn_out)
        tag_space = self.hidden2tag(rnn_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

