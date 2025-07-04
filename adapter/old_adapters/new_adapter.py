import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# sasika edited this best v4; 
class DCTAdapter(nn.Module):
    def __init__(self, input_dim=768, num_components=24, tau=1.0):
        super().__init__()
        self.tau = tau
        self.adapter_gate_logits = nn.Parameter(torch.randn(input_dim))  # One per DCT dim
        self.adapter_down = nn.Linear(input_dim, 18, bias=False) # 18 best
        self.adapter_up = nn.Linear(18, input_dim, bias=False) # 18 best

    def gumbel_softmax_mask(self, logits):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        y = logits + gumbel_noise
        return F.softmax(y / self.tau, dim=-1)  # soft but approximates hard gate

    def dct1(self, x):  # same as before
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat.T

    def idct1(self, x):
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat

    def create_dct_matrix(self, N, device=None, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / N)
        return dct

    def forward(self, hidden_states):
        dct = self.dct1(hidden_states)  # [B, T, C]

        # gate_mask = self.gumbel_softmax_mask(self.adapter_gate_logits)  # [C]
        # gated_dct = dct * gate_mask  # broadcasted over B and T

        z = dct.reshape(-1, dct.shape[-1])  # [B*T, C]
        z_pert = self.adapter_up(F.tanh(self.adapter_down(z)))
        out = z_pert.view_as(dct)
        idct = self.idct1(out)
        return hidden_states + idct  # residual connection


