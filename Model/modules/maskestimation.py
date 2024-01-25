import torch
import torch.nn as nn

class GLU(nn.Module):
    """
    GLU Activation Module.
    """
    def __init__(self, input_dim ):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x[..., :self.input_dim] * self.sigmoid(x[..., self.input_dim:])
        return x


class MLP(nn.Module):
    """
    Just a simple MLP with tanh activation (by default).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation_type = 'tanh',):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim)
        )

    @staticmethod
    def select_activation(activation_type: str) -> nn.modules.activation:
        if activation_type == 'tanh':
            return nn.Tanh
        elif activation_type == 'relu':
            return nn.ReLU
        elif activation_type == 'gelu':
            return nn.GELU
        else:
            raise ValueError("wrong activation function was selected")

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class MaskEstimationModule(nn.Module):
    """
    MaskEstimation (3rd) Module of BandSplitRNN.
    Recreates from input initial subband dimensionality via running through LayerNorms+MLPs and forms the T-F mask.
    """

    def __init__(self, bands, t_timesteps  = 517, fc_dim  = 128, mlp_dim  = 512, complex_as_channel = True, is_mono = False):
        super(MaskEstimationModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.frequency_mul = frequency_mul

        self.bandwidths = [(e - s) for s, e in bands]
        self.layernorms = nn.ModuleList([
            nn.LayerNorm([t_timesteps, fc_dim])
            for _ in range(len(self.bandwidths))
        ])
        self.mlp = nn.ModuleList([
            MLP(fc_dim, mlp_dim, bw * frequency_mul, activation_type='tanh')
            for bw in self.bandwidths
        ])

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, k_subbands, time, fc_shape]
        Output: [batch_size, freq, time]
        """
        outs = []
        for i in range(x.shape[1]):
            # run through model
            out = self.layernorms[i](x[:, i])
            out = self.mlp[i](out)
            B, T, F = out.shape
            # return to complex
            if self.cac:
                out = out.permute(0, 2, 1).contiguous()
                out = out.view(B, -1, 2, F//self.frequency_mul, T).permute(0, 1, 3, 4, 2)
                out = torch.view_as_complex(out.contiguous())
            else:
                out = out.view(B, -1, F//self.frequency_mul, T).contiguous()
            outs.append(out)

        # concat all subbands
        outs = torch.cat(outs, dim=-2)
        return outs