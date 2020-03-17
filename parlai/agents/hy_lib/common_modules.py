import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_sizes=(512,),
                 activation="Tanh", bias=True, dropout=0.1):
        super(FeedForward, self).__init__()
        self.activation = getattr(nn, activation)()

        n_inputs = [input_dim] + list(hidden_sizes)
        n_outputs = list(hidden_sizes) + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])
        self.num_layer = len(self.linears)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_):
        x = input_
        i = 0
        for linear in self.linears:
            x = linear(x)
            if i < self.num_layer - 1:
                x = self.dropout_layer(x)
            x = self.activation(x)
            i += 1
        return x
