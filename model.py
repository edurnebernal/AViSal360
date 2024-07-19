import modules
import torch.nn as nn
import torch
from spherenet import SphereConv2D

class AVSal_IB4LconAEM_2SP(nn.Module):
    # TODO: test multipliying instead of concatenating
    def __init__(self, input_dim=3, hidden_dim=18, output_dim=1, option='train'):
        super(AVSal_IB4LconAEM_2SP, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)
        
        self.mode = option
        # Create two dense layers to reduce the embedding dimensionality to 32
        self.audio_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid()
        )

        self.av_layers = nn.Sequential(
                # Add a spherical convolutional layer
                SphereConv2D(hidden_dim*2+1, hidden_dim*2, bias=True),
                nn.BatchNorm2d(hidden_dim*2),
                nn.LeakyReLU(),
                # Add a spherical convolutional layer
                SphereConv2D(hidden_dim*2, hidden_dim, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU()
            )

            

    def forward(self, img, aem, emb):
 
        b, _, _, h, w = img.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))

        # Eliminate second dimension of the embedding
        f_aud = self.audio_layers(emb.squeeze(1).squeeze(1))
        f_aud = f_aud.unsqueeze(-1).unsqueeze(-1)
        f_aud = f_aud.repeat(1, 1, 120, 160)    

        # Process the frames sequence with the ConvLSTM encoder
        for t in range(img.shape[1]):
            f_vis, state_e = self.encoder(img[:, t, :, :, :], state_e)

            # Repeat the aem per num channels
            expanded_aem = aem[:, t, :, :, :]
                
            audio = torch.cat((f_aud, expanded_aem), dim=1)

            # Concate the audio features with the visual features
            f_av = torch.cat((f_vis, audio), dim=1)
            f_av = self.av_layers(f_av)
            out, state_d = self.decoder(f_av, state_d)
            
            return out
