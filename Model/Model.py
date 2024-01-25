import torch
import torch.nn as nn
import scipy.signal as signal

from modules import bandsplit, modeling, maskestimation

class BandSplitRNN(nn.module):

    def __init__(self, duration, bands, complex, mono, nndim, hidden_dim_size, returnMask, stems):
        super(BandSplitRNN, self).__init__()

        self.bandsplit = bandsplit(duration, bands, complex, mono, nndim)
        self.BandSequence = modeling(nndim, hidden_dim_size)
        self.MaskEstimation = maskestimation(bands)

        self.complex = complex
        self.mask = returnMask
        self.stems = stems

    def wiener(self, x_hat, x_complex):
        """
        Wiener filtering of the input signal
        """
        # TODO: add Wiener Filtering

        x_hat = signal.wiener(x_hat)

        return x_hat

    def compute_mask(self, x):
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        ret = []
        for _ in self.stems:
            ret.append(self.maskest(x))  # [batch_size, freq, time]

        return ret
    
    def forward(self, x):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        # use only magnitude if not using complex input
        x_complex = None
        if not self.cac:
            x_complex = x
            x = x.abs()
        # normalize
        # TODO: Try to normalize in bandsplit and denormalize in maskest
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # compute T-F mask
        mask = self.compute_mask(x)

        # multiply with original tensor
        x = mask if self.return_mask else mask * x

        # denormalize
        x = x * std + mean

        if not self.cac:
            x = self.wiener(x, x_complex)

        return x