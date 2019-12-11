import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, num_hidden):

        super(AE, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = latent_dim
        self.num_hidden = num_hidden

        self.encoder = nn.ModuleList()
        encoder_dims = [input_dim] + [hidden_dim]*num_hidden + [latent_dim]
        for i in range(len(encoder_dims)-1):
            self.encoder.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))

        self.decoder = nn.ModuleList()
        decoder_dims = [latent_dim] + [hidden_dim]*num_hidden + [input_dim]
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))

    def forward(self, x):
        # encoder
        for i in range(self.num_hidden):
            x = F.relu(self.encoder[i](x))
        x = self.encoder[self.num_hidden](x)  # ToDo ReLu or Identity?

        # decoder
        for i in range(self.num_hidden):
            x = F.relu(self.decoder[i](x))
        x = self.decoder[self.num_hidden](x)

        return x
