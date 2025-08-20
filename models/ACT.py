# Action Chunking with Transformers
# https://arxiv.org/pdf/2304.13705

import torch
import torch.nn as nn
from config import HIDDEN_DIM, LATENT_DIM, CHUNK_SIZE, ACTION_DIM

# Training: CVAEencoder + CVAEDecoder (PolicyEncoder and PolicyDecoder)
# Testing: CVAEDecoder (PolicyEncoder and PolicyDecoder)

class CVAE_encoder(nn.Module):
    '''
    Computes the parameters of the latent variable z's distribution.
    Only used in training 

    Input: 
        - Embedded target action sequence (k actions)
        - Embedded joints
        - [CLS] token

    Output:
        - z (the latent "style" variable)
    '''
    def __init__(self):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(CHUNK_SIZE * ACTION_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        # mu and log var will be used to sample latent variable z
        self.mu_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.log_var_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, actions, qpos):
        x = torch.cat([actions.flatten(1), qpos], dim=1)
        encoded = self.encoder_net(x)
        mu = self.mu_head(encoded)
        log_var = self.log_var_head(encoded)
        return mu, log_var



class PolicyEncoder(nn.Module):
    """
    Processes observations and creates a contextual memory
    Input:
        - Embedded camera footage
        - Embedded joints
        - z (z in training, in test we give 0)

    Output:
        - Synthesized embedding from input
    """
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, src):
        return self.transformer_encoder(src)


class PolicyDecoder(nn.Module):
    """
    Generates the action sequence based on the output from the PolicyEncoder.
    Input: 
        - Encoder output

    Output:
        - Predicted next k actions
    """
    def __init__(self):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=HIDDEN_DIM, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, CHUNK_SIZE, HIDDEN_DIM))
        self.action_head = nn.Linear(HIDDEN_DIM, ACTION_DIM)

    def forward(self, memory):
        batch_size = memory.shape[0]
        tgt = self.decoder_pos_embed.expand(batch_size, -1, -1)
        output = self.transformer_decoder(tgt=tgt, memory=memory)
        pred_actions = self.action_head(output)
        return pred_actions



