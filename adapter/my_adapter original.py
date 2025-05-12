import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TinyAutoEncoderAdapter(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=32):
        super().__init__()

        # Design: 768 → 64 → 32 → 64 → 768
        # Total params = 768*64 + 64*32 + 32*64 + 64*768 = 49,152

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32, bias=False),    # 768 → 64
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim, bias=False),  # 64 → 32
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32, bias=False),  # 32 → 64
            nn.ReLU(),
            nn.Linear(32, input_dim, bias=False)     # 64 → 768
        )

    def forward(self, x):
        # x: [B, T, D]
        shape = x.shape
        x_flat = x.view(-1, shape[-1])  # [B*T, D]
        z = self.encoder(x_flat)
        out = self.decoder(z)
        return out.view(*shape)  # Reshape to [B, T, D]

# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim,num_components=24):
#         super(DCTAdapter, self).__init__()
        
#         self.num_components = num_components
#         # self.adapter = nn.Linear(self.num_components,self.num_components)
#         self.adapter = nn.Parameter(torch.zeros(self.num_components))
#     def dct1(self,x):
#         # x: [B, T, H] — apply DCT over last dim (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

#     def idct1(self,x):
#         # x: [B, T, H] — apply inverse DCT over last dim
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

#     def create_dct_matrix(self,N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]

#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct  # shape: [N, N]
#     def whiten_and_recolor(self, z_pert, z_orig, eps=1e-5):
#         """
#         z_pert: [batch_size, latent_dim] — perturbed latent vectors
#         z_orig: [batch_size, latent_dim] — original latent vectors (reference)
#         """
#         # Center original data
#         mu = z_orig.mean(dim=0, keepdim=True)  # [1, latent_dim]
#         z_centered = z_orig - mu               # [batch, latent_dim]

#         # Compute covariance matrix [latent_dim x latent_dim]
#         cov = (z_centered.T @ z_centered) / (z_orig.shape[0] - 1) + eps * torch.eye(z_orig.shape[1], device=z_orig.device)

#         # SVD for square root and inverse square root
#         U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
#         S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
#         S_sqrt = torch.diag(torch.sqrt(S))

#         cov_inv_sqrt = U @ S_inv_sqrt @ U.T  # C^(-1/2)
#         cov_sqrt = U @ S_sqrt @ U.T          # C^(1/2)

#         # Center perturbed
#         z_pert_centered = z_pert - mu

#         # Whitening
#         z_pert_white = z_pert_centered @ cov_inv_sqrt

#         # Coloring
#         z_pert_recolored = z_pert_white @ cov_sqrt + mu

#         return z_pert_recolored

#     def forward(self, hidden_states):
#         # print('hidden_states: ',hidden_states.shape)
#         hidden_states_dct = self.dct1(hidden_states)
#         # print('DCT: ',hidden_states_dct.shape)
#         hidden_states_kdims = hidden_states_dct[:,:,:self.num_components]
#         z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         # print('Z: ',z.shape)
#         # print(self.adapter)
#         # z_pred1 = self.adapter(z)
#         z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
#         # print(z_pred)
#         # z_pred = F.relu(z-z.mean(dim=0, keepdim=True)) + z
#         # z_pred = z + self.adapter*(z-z.mean(dim=0, keepdim=True))
#         z_pred = z_pred.reshape(hidden_states_kdims.shape)
#         hidden_states_dct[:,:,:self.num_components] = z_pred
#         reconstructed_hidden_states = self.idct1(hidden_states_dct) #+ hidden_states
#         # z_pert_out = self.whiten_and_recolor(z_pert,z)
#         # print('reconstructed: ',reconstructed_hidden_states.shape)
#         return reconstructed_hidden_states

class DCTAttentionAdapter(nn.Module):
    def __init__(self, input_dim, num_components=24, heads=2):
        super(DCTAttentionAdapter, self).__init__()
        self.num_components = num_components
        self.heads = heads
        self.input_dim = input_dim

        # Simple attention in DCT space
        self.adapter_q_proj = nn.Linear(num_components, num_components)
        self.adapter_k_proj = nn.Linear(num_components, num_components)
        self.adapter_v_proj = nn.Linear(num_components, num_components)
        self.adapter_out_proj = nn.Linear(num_components, num_components)

    def create_dct_matrix(self, N, device=None, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / N)
        return dct

    def dct1(self, x):
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)
        return x @ dct_mat.T

    def idct1(self, x):
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)
        return x @ dct_mat

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]
        B, T, D = hidden_states.shape
        hidden_states_dct = self.dct1(hidden_states)  # [B, T, D]

        top_k = hidden_states_dct[:, :, :self.num_components]  # [B, T, K]
        z = top_k.reshape(-1, self.num_components).unsqueeze(1)  # [B*T, 1, K]

        # Attention: simple self-attention over length-1 sequence (i.e., token-wise)
        q = self.adapter_q_proj(z)  # [B*T, 1, K]
        k = self.adapter_k_proj(z)
        v = self.adapter_v_proj(z)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.num_components), dim=-1)
        z_attn = attn @ v  # [B*T, 1, K]
        z_attn = self.adapter_out_proj(z_attn)  # [B*T, 1, K]
        z_pred = z.squeeze(1) + z_attn.squeeze(1)  # residual

        # Inject back the perturbed DCT coefficients
        hidden_states_dct_clone = hidden_states_dct.clone()
        hidden_states_dct_clone[:, :, :self.num_components] = z_pred.view(B, T, -1)

        # Inverse DCT
        out = self.idct1(hidden_states_dct_clone) + hidden_states
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HadamardAdapter(nn.Module):
    def __init__(self, input_dim, num_components=24):
        super(HadamardAdapter, self).__init__()
        
        self.original_dim = input_dim
        self.padded_dim = 2 ** math.ceil(math.log2(input_dim))
        self.num_components = num_components

        # Learnable perturbation weights for top-k components
        self.adapter = nn.Parameter(torch.zeros(self.num_components))

        # Register Hadamard matrix as buffer
        self.register_buffer("H", self._create_hadamard(self.padded_dim))

    def _create_hadamard(self, n):
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H,  H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0)
        H = H / math.sqrt(n)
        return H  # [n, n]

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]
        B, T, D = hidden_states.shape
        assert D == self.original_dim

        # Pad if needed
        if D < self.padded_dim:
            pad_size = self.padded_dim - D
            hidden_states = F.pad(hidden_states, (0, pad_size))  # [B, T, padded_dim]

        H = self.H.to(hidden_states.device, hidden_states.dtype)
        hidden_h = hidden_states @ H.T  # Hadamard forward: [B, T, padded_dim]

        top_k = hidden_h[:, :, :self.num_components]  # [B, T, K]
        z = top_k.reshape(-1, self.num_components)    # [B*T, K]

        # Apply learnable perturbation
        z_pert = z + F.leaky_relu(self.adapter) * (z - z.mean(dim=0, keepdim=True))

        # Replace perturbed components
        hidden_h = hidden_h.clone()  # avoid in-place modification
        hidden_h[:, :, :self.num_components] = z_pert.view(B, T, -1)

        # Inverse Hadamard
        out = hidden_h @ H  # [B, T, padded_dim]

        # Truncate to original dim
        out = out[:, :, :self.original_dim]
        return out


#best
# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim,num_components=24):
#         super(DCTAdapter, self).__init__()
        
#         self.num_components = num_components
#         # self.adapter = nn.Linear(self.num_components,self.num_components)
#         self.adapter_down = nn.Linear(self.num_components, 256, bias=False) #best
#         self.adapter_up = nn.Linear(256, self.num_components, bias=False) #best
#         # self.adapter_down = nn.Linear(self.num_components, 64, bias=True)
#         # self.adapter_up = nn.Linear(64, self.num_components, bias=True) 
#     def dct1(self,x):
#         # x: [B, T, H] — apply DCT over last dim (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

#     def idct1(self,x):
#         # x: [B, T, H] — apply inverse DCT over last dim
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

#     def create_dct_matrix(self,N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]

#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct  # shape: [N, N]
#     def whiten_and_recolor(self, z_pert, z_orig, eps=1e-5):
#         """
#         z_pert: [batch_size, latent_dim] — perturbed latent vectors
#         z_orig: [batch_size, latent_dim] — original latent vectors (reference)
#         """
#         # Center original data
#         mu = z_orig.mean(dim=0, keepdim=True)  # [1, latent_dim]
#         z_centered = z_orig - mu               # [batch, latent_dim]

#         # Compute covariance matrix [latent_dim x latent_dim]
#         cov = (z_centered.T @ z_centered) / (z_orig.shape[0] - 1) + eps * torch.eye(z_orig.shape[1], device=z_orig.device)

#         # SVD for square root and inverse square root
#         U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
#         S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
#         S_sqrt = torch.diag(torch.sqrt(S))

#         cov_inv_sqrt = U @ S_inv_sqrt @ U.T  # C^(-1/2)
#         cov_sqrt = U @ S_sqrt @ U.T          # C^(1/2)

#         # Center perturbed
#         z_pert_centered = z_pert - mu

#         # Whitening
#         z_pert_white = z_pert_centered @ cov_inv_sqrt

#         # Coloring
#         z_pert_recolored = z_pert_white @ cov_sqrt + mu

#         return z_pert_recolored

#     def forward(self, hidden_states):
#         # print('hidden_states: ',hidden_states.shape)
#         hidden_states_dct = self.dct1(hidden_states)
#         # print('DCT: ',hidden_states_dct.shape)
#         hidden_states_kdims = hidden_states_dct[:,:,:self.num_components]
#         # z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         # print('Z: ',z.shape)
#         # print(self.adapter)
#         # z_pred1 = self.adapter(z)
        
#         # z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
#         # z_pred = z_pred.reshape(hidden_states_kdims.shape)

#         z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         z_pert = self.adapter_up(F.relu(self.adapter_down(z)))
#         z_pred = z_pert

#         hidden_states_dct_clone = hidden_states_dct.clone()
#         hidden_states_dct_clone[:, :, :self.num_components] = z_pred
#         reconstructed_hidden_states = self.idct1(hidden_states_dct_clone) + hidden_states

#         # hidden_states_dct[:,:,:self.num_components] = z_pred
#         # reconstructed_hidden_states = self.idct1(hidden_states_dct) #hidden_states
#         # z_pert_out = self.whiten_and_recolor(z_pert,z)
#         # print('reconstructed: ',reconstructed_hidden_states.shape)
#         return reconstructed_hidden_states

#best v2
# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim,num_components=24):
#         super(DCTAdapter, self).__init__()
        
#         self.num_components = num_components
#         # self.adapter = nn.Linear(self.num_components,self.num_components)
#         self.adapter_down = nn.Linear(768, 64, bias=False) #best is 32
#         self.adapter_up = nn.Linear(64, 768, bias=False) #best is 32
#         # self.adapter_down = nn.Linear(self.num_components, 64, bias=True)
#         # self.adapter_up = nn.Linear(64, self.num_components, bias=True) 
#     def dct1(self,x):
#         # x: [B, T, H] — apply DCT over last dim (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

#     def idct1(self,x):
#         # x: [B, T, H] — apply inverse DCT over last dim
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

#     def create_dct_matrix(self,N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]

#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct  # shape: [N, N]
#     def whiten_and_recolor(self, z_pert, z_orig, eps=1e-5):
#         """
#         z_pert: [batch_size, latent_dim] — perturbed latent vectors
#         z_orig: [batch_size, latent_dim] — original latent vectors (reference)
#         """
#         # Center original data
#         mu = z_orig.mean(dim=0, keepdim=True)  # [1, latent_dim]
#         z_centered = z_orig - mu               # [batch, latent_dim]

#         # Compute covariance matrix [latent_dim x latent_dim]
#         cov = (z_centered.T @ z_centered) / (z_orig.shape[0] - 1) + eps * torch.eye(z_orig.shape[1], device=z_orig.device)

#         # SVD for square root and inverse square root
#         U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
#         S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
#         S_sqrt = torch.diag(torch.sqrt(S))

#         cov_inv_sqrt = U @ S_inv_sqrt @ U.T  # C^(-1/2)
#         cov_sqrt = U @ S_sqrt @ U.T          # C^(1/2)

#         # Center perturbed
#         z_pert_centered = z_pert - mu

#         # Whitening
#         z_pert_white = z_pert_centered @ cov_inv_sqrt

#         # Coloring
#         z_pert_recolored = z_pert_white @ cov_sqrt + mu

#         return z_pert_recolored

#     def forward(self, hidden_states):
#         # print('hidden_states: ',hidden_states.shape)
#         # hidden_states_dct = self.dct1(hidden_states)
#         hidden_states_dct = hidden_states
#         # print('DCT: ',hidden_states_dct.shape)
#         hidden_states_kdims = hidden_states_dct#[:,:,:self.num_components]
#         # z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         # print('Z: ',z.shape)
#         # print(self.adapter)
#         # z_pred1 = self.adapter(z)
        
#         # z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
#         # z_pred = z_pred.reshape(hidden_states_kdims.shape)

#         z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
#         z_pred = z_pert

#         # hidden_states_dct_clone = hidden_states_dct.clone()
#         # hidden_states_dct_clone[:, :, :self.num_components] = z_pred
#         reconstructed_hidden_states = self.idct1(z_pred) + hidden_states
#         reconstructed_hidden_states = z_pred + hidden_states

#         # hidden_states_dct[:,:,:self.num_components] = z_pred
#         # reconstructed_hidden_states = self.idct1(hidden_states_dct) #hidden_states
#         # z_pert_out = self.whiten_and_recolor(z_pert,z)
#         # print('reconstructed: ',reconstructed_hidden_states.shape)
#         return reconstructed_hidden_states

# sasika edited the following
class DCTAdapter(nn.Module):
    def __init__(self, input_dim,num_components=24):
        super(DCTAdapter, self).__init__()
        
        self.num_components = num_components
        # self.adapter = nn.Linear(self.num_components,self.num_components)
        self.adapter_down = nn.Linear(768, 16, bias=False) #best is 32
        self.adapter_up = nn.Linear(16, 768, bias=False) #best is 32
        # self.adapter_down = nn.Linear(self.num_components, 64, bias=True)
        # self.adapter_up = nn.Linear(64, self.num_components, bias=True) 
    def dct1(self,x):
        # x: [B, T, H] — apply DCT over last dim (H)
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
        return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

    def idct1(self,x):
        # x: [B, T, H] — apply inverse DCT over last dim
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
        return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

    def create_dct_matrix(self,N, device=None, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]

        dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / N)
        return dct  # shape: [N, N]

    def forward(self, hidden_states):
        # print('hidden_states: ',hidden_states.shape)
        # hidden_states_dct = self.dct1(hidden_states)
        hidden_states_dct = hidden_states
        # print('DCT: ',hidden_states_dct.shape)
        hidden_states_kdims = hidden_states_dct#[:,:,:self.num_components]
        # z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
        # print('Z: ',z.shape)
        # print(self.adapter)
        # z_pred1 = self.adapter(z)
        
        # z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
        # z_pred = z_pred.reshape(hidden_states_kdims.shape)

        z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
        z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
        z_pred = z_pert

        # hidden_states_dct_clone = hidden_states_dct.clone()
        # hidden_states_dct_clone[:, :, :self.num_components] = z_pred
        reconstructed_hidden_states = self.idct1(z_pred) + hidden_states
        reconstructed_hidden_states = z_pred + hidden_states

        # hidden_states_dct[:,:,:self.num_components] = z_pred
        # reconstructed_hidden_states = self.idct1(hidden_states_dct) #hidden_states
        # z_pert_out = self.whiten_and_recolor(z_pert,z)
        # print('reconstructed: ',reconstructed_hidden_states.shape)
        return reconstructed_hidden_states


# Original as of 08-05-2025
# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim,num_components=24):
#         super(DCTAdapter, self).__init__()
        
#         self.num_components = num_components
#         # self.adapter = nn.Linear(self.num_components,self.num_components)
#         self.adapter_down = nn.Linear(768, 16, bias=False) #best is 32
#         self.adapter_up = nn.Linear(16, 768, bias=False) #best is 32
#         # self.adapter_down = nn.Linear(self.num_components, 64, bias=True)
#         # self.adapter_up = nn.Linear(64, self.num_components, bias=True) 
#     def dct1(self,x):
#         # x: [B, T, H] — apply DCT over last dim (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

#     def idct1(self,x):
#         # x: [B, T, H] — apply inverse DCT over last dim
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

#     def create_dct_matrix(self,N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]

#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct  # shape: [N, N]

#     def forward(self, hidden_states):
#         # print('hidden_states: ',hidden_states.shape)
#         # hidden_states_dct = self.dct1(hidden_states)
#         hidden_states_dct = hidden_states
#         # print('DCT: ',hidden_states_dct.shape)
#         hidden_states_kdims = hidden_states_dct#[:,:,:self.num_components]
#         # z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         # print('Z: ',z.shape)
#         # print(self.adapter)
#         # z_pred1 = self.adapter(z)
        
#         # z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
#         # z_pred = z_pred.reshape(hidden_states_kdims.shape)

#         z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
#         z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
#         z_pred = z_pert

#         # hidden_states_dct_clone = hidden_states_dct.clone()
#         # hidden_states_dct_clone[:, :, :self.num_components] = z_pred
#         reconstructed_hidden_states = self.idct1(z_pred) + hidden_states
#         reconstructed_hidden_states = z_pred + hidden_states

#         # hidden_states_dct[:,:,:self.num_components] = z_pred
#         # reconstructed_hidden_states = self.idct1(hidden_states_dct) #hidden_states
#         # z_pert_out = self.whiten_and_recolor(z_pert,z)
#         # print('reconstructed: ',reconstructed_hidden_states.shape)
#         return reconstructed_hidden_states

# best v3
# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim=768, num_components=24, tau=1.0):
#         super().__init__()
#         self.tau = tau
#         self.adapter_gate_logits = nn.Parameter(torch.randn(input_dim))  # One per DCT dim
#         self.adapter_down = nn.Linear(input_dim, 24, bias=False)
#         self.adapter_up = nn.Linear(24, input_dim, bias=False)

#     def gumbel_softmax_mask(self, logits):
#         gumbel_noise = -torch.empty_like(logits).exponential_().log()
#         y = logits + gumbel_noise
#         return F.softmax(y / self.tau, dim=-1)  # soft but approximates hard gate

#     def dct1(self, x):  # same as before
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
#         return x @ dct_mat.T

#     def idct1(self, x):
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
#         return x @ dct_mat

#     def create_dct_matrix(self, N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct

#     def forward(self, hidden_states):
#         dct = self.dct1(hidden_states)  # [B, T, C]

#         # gate_mask = self.gumbel_softmax_mask(self.adapter_gate_logits)  # [C]
#         # gated_dct = dct * gate_mask  # broadcasted over B and T

#         z = dct.reshape(-1, dct.shape[-1])  # [B*T, C]
#         z_pert = self.adapter_up(F.relu(self.adapter_down(z)))
#         out = z_pert.view_as(dct)
#         idct = self.idct1(out)
#         return hidden_states + idct  # residual connection

#best v4
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
        z_pert = self.adapter_up(F.leaky_relu(self.adapter_down(z)))
        out = z_pert.view_as(dct)
        idct = self.idct1(out)
        return hidden_states + idct  # residual connection




# class DCTAdapter(nn.Module):
#     def __init__(self, input_dim=768, num_components=24, output_dim=768, n_filters=8):
#         super(DCTAdapter, self).__init__()
        
#         # Depth-wise and Point-wise adapters
#         self.adapter_depthwise = nn.Linear(input_dim, n_filters, bias=False)
#         self.adapter_pointwise = nn.Linear(n_filters, output_dim, bias=False)
        
#         # Number of frequency components for DCT
#         self.num_components = num_components

#     def create_dct_matrix(self, N, device=None, dtype=torch.float32):
#         n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # shape: [1, N]
#         k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # shape: [N, 1]
#         dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # shape: [N, N]
#         dct[0] *= 1 / math.sqrt(2)
#         dct *= math.sqrt(2 / N)
#         return dct  # shape: [N, N]
    
#     def dct1(self, x):
#         # Apply DCT on the last dimension (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat.T  # [B, T, H] x [H, H] → [B, T, H]

#     def idct1(self, x):
#         # Apply inverse DCT on the last dimension (H)
#         N = x.size(-1)
#         dct_mat = self.create_dct_matrix(N, device=x.device, dtype=x.dtype)  # [N, N]
#         return x @ dct_mat  # [B, T, H] x [H, H] → [B, T, H]

#     def forward(self, hidden_states):
#         B, T, D = hidden_states.shape
        
#         # Step 1: Apply DCT to convert hidden states to frequency domain
#         hidden_states_dct = self.dct1(hidden_states)  # [B, T, D]
        
#         # Step 2: Extract the most important frequency components
#         hidden_states_kdims = hidden_states_dct#[:, :, :self.num_components]  # [B, T, num_components]
        
#         # Flatten for the depth-wise adapter
#         z = hidden_states_kdims.view(-1, hidden_states_kdims.shape[-1])  # [B*T, num_components]
        
#         # Step 3: Apply depth-wise and point-wise adapters
#         z_depthwise = F.leaky_relu(self.adapter_depthwise(z))  # [B*T, n_filters]
#         z_adapted = self.adapter_pointwise(z_depthwise)  # [B*T, D]
        
#         # Reshape back to original shape
#         z_adapted = z_adapted.view(B, T, D)
        
#         # Step 4: Add adapted features to the original hidden states (residual connection)
#         adapted_hidden_states = hidden_states_dct + z_adapted  # [B, T, D]
        
#         # Step 5: Apply inverse DCT to get back to the original space
#         reconstructed_hidden_states = self.idct1(adapted_hidden_states)  # [B, T, D]
        
#         return reconstructed_hidden_states