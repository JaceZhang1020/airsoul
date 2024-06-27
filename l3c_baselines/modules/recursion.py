import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        if(cache is None):
            lstm_out, new_cache = self.lstm(src)
        else:
            lstm_out, new_cache = self.lstm(src, cache)
        output = self.fc(lstm_out)
        return output, new_cache

class PRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Plastic Parameters
        self.Wih_a = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_b = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.Wih_d = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bih = nn.Parameter(torch.Tensor(hidden_size))

        # Dopamine gates
        self.dopamine = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # 初始化参数
        self.reset_parameters()
        self.alpha = 1.0e-3

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wih_a)
        nn.init.xavier_uniform_(self.Wih_b)
        nn.init.xavier_uniform_(self.Wih_c)
        nn.init.xavier_uniform_(self.Wih_d)
        nn.init.zeros_(self.bih)

    def init_memories(self, batch_size, device):
        wih = torch.zeros(batch_size, self.hidden_size, self.input_size + self.hidden_size)
        h = torch.zeros(batch_size, self.hidden_size)
        return h.to(device), wih.to(device)

    def forward(self, inputs, memories=None):
        if(memories is not None):
            hidden, wih = memories
        else:
            hidden, wih = self.init_memories(inputs.shape[0], inputs.device)

        x = torch.cat((inputs, hidden), 1)

        dopamine_signal = self.dopamine(x)

        new_hidden = F.tanh(torch.einsum('ij,ikj->ik', x, wih) + self.bih)

        x = x.unsqueeze(1)
        x_ = torch.ones_like(x).to(inputs.device)
        y = new_hidden.unsqueeze(2)
        y_ = torch.ones_like(y).to(inputs.device)

        wih = wih + self.alpha * (torch.bmm(y, x) * self.Wih_a +
                torch.bmm(y_, x) * self.Wih_b +
                torch.bmm(y, x_) * self.Wih_c +
                torch.bmm(y_, x_) * self.Wih_d)

        return new_hidden, wih

class PRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PRNN, self).__init__()
        self.hidden_size = hidden_size
        self.prnn_cell = PRNNCell(input_size, hidden_size) 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        if(cache is None):
            h, W = self.prnn_cell.init_memories(src.shape[0], src.device)
        else:
            h, W = cache
        hiddens = []
        for i in range(src.shape[1]):
            h, W = self.prnn_cell(src[:, i], (h, W))
            hiddens.append(h.unsqueeze(1))
        outputs = self.fc(torch.cat(hiddens, dim=1))
        new_cache = (h, W)
        return outputs, new_cache

if __name__=='__main__':
    inputs = torch.randn(4, 8, 64)
    model = PRNN(64, 128, 64)
    outputs, mems = model(inputs)
    print(outputs.shape, mems[0].shape, mems[1].shape)
