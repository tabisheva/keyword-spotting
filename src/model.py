import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(self.fc(x))
        alpha = torch.softmax(outputs, dim=1)
        x = (x * alpha).sum(dim=1)
        return x

class KWSNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.cnn = nn.Sequential(
            nn.Conv1d(self.params["num_features"], self.params["cnn_channels"],
                      kernel_size=self.params["cnn_kernel_size"], padding=self.params["cnn_kernel_size"] // 2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=self.params["cnn_channels"], hidden_size=self.params["gru_hidden_size"],
                          bidirectional=True, batch_first=True)
        self.attention = Attention(self.params["gru_hidden_size"] * 2, self.params["attention_hidden_size"])
        self.linear = nn.Linear(self.params["gru_hidden_size"] * 2, 1 + 1, bias=False) # 1 keyword


    def forward(self, x):
        conv = self.cnn(x).permute(0, 2, 1)
        rnn_output, _ = self.rnn(conv)
        linear_attn = self.linear(self.attention(rnn_output))
        return torch.log_softmax(linear_attn, dim=1)

    def inference(self, x: torch.Tensor, window_size: int):
        if window_size > x.shape[2]:
            window_size = x.shape[2]
        probs = []
        hidden = None
        for i in range(window_size, x.shape[2] + 1, 50):
            window = x[:, :, i - window_size:i]
            window = self.cnn(window)
            window = window.permute(0, 2, 1)
            window, h = self.rnn(window, hidden)
            window = self.attention(window)
            window = self.linear(window)
            p = torch.softmax(window, dim=1).squeeze()[1]
            probs.append(p.item())
        return probs
