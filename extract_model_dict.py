import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        s0, s1, s2 = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(s0, s1 * s2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(s0, s1, s2)
        return x

lstm = Autoencoder(input_dim=300*7, hidden_dim=64)
lstm.load_state_dict(torch.load('./LSTM-prediction_20_300_2_64_64_biaozhun_model.pt'))

for param_tensor in lstm.state_dict():
    with open(f'{param_tensor}.txt', 'w') as f:
        if(len(lstm.state_dict()[param_tensor].size())>1):
            for row in lstm.state_dict()[param_tensor]:
                f.write('\t'.join([str(value.item()) for value in row]) + '\n')
        else:
            for value in lstm.state_dict()[param_tensor]:
                f.write(f"{value.item()}\n")
