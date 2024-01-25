import torch
import torch.nn as nn

class BandSplit(nn.module):
    def __init__(self, duration, bands, complex, mono, nndim):
        super(BandSplit, self).__init__()

        self.bands = bands
        
        if complex or not mono:
            frequency_multiplier = 2
        else:
            frequency_multiplier = 1

        self.norm = nn.ModuleList([
            nn.LayerNorm([(end - start) * frequency_multiplier, duration])
            for start, end in bands
        ])

        self.fc = nn.ModuleList([
            nn.Linear((end- start)* frequency_multiplier, nndim)
            for start, end in bands
        ])

        def splitInSubbands(self, x):
            subbands = []
            for band in self.bands:
                subbands.append(x[band[0]:band[1]])
            return subbands
        
        def forward(self, x):
            processed = []
            i = 0
            for subband in splitInSubbands(x):
                Batch, Chan, Freq, Time = x.shape
                if x.dtype == torch.cfloat:
                    x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
                # from channels to frequency
                x = x.reshape(Batch, -1, Time)
                
                x = self.norm[i](x)
                x = x.transpose(-1, -2)
                x = self.fc[i](x)
                processed.append(x)
                i+=1
            return torch.stack(processed, dim=1)                            
