import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

class DCTAdapter(nn.Module):
    def __init__(self, input_dim,num_components=24):
        super(DCTAdapter, self).__init__()
        
        self.num_components = num_components
        # self.adapter = nn.Linear(self.num_components,self.num_components)
        self.adapter_down = nn.Linear(self.num_components, 256, bias=False)
        self.adapter_up = nn.Linear(256, self.num_components, bias=False) 
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
    def whiten_and_recolor(self, z_pert, z_orig, eps=1e-5):
        """
        z_pert: [batch_size, latent_dim] — perturbed latent vectors
        z_orig: [batch_size, latent_dim] — original latent vectors (reference)
        """
        # Center original data
        mu = z_orig.mean(dim=0, keepdim=True)  # [1, latent_dim]
        z_centered = z_orig - mu               # [batch, latent_dim]

        # Compute covariance matrix [latent_dim x latent_dim]
        cov = (z_centered.T @ z_centered) / (z_orig.shape[0] - 1) + eps * torch.eye(z_orig.shape[1], device=z_orig.device)

        # SVD for square root and inverse square root
        U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
        S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
        S_sqrt = torch.diag(torch.sqrt(S))

        cov_inv_sqrt = U @ S_inv_sqrt @ U.T  # C^(-1/2)
        cov_sqrt = U @ S_sqrt @ U.T          # C^(1/2)

        # Center perturbed
        z_pert_centered = z_pert - mu

        # Whitening
        z_pert_white = z_pert_centered @ cov_inv_sqrt

        # Coloring
        z_pert_recolored = z_pert_white @ cov_sqrt + mu

        return z_pert_recolored

    def forward(self, hidden_states):
        # print('hidden_states: ',hidden_states.shape)
        hidden_states_dct = self.dct1(hidden_states)
        # print('DCT: ',hidden_states_dct.shape)
        hidden_states_kdims = hidden_states_dct[:,:,:self.num_components]
        # z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
        # print('Z: ',z.shape)
        # print(self.adapter)
        # z_pred1 = self.adapter(z)
        
        # z_pred = z + F.leaky_relu(self.adapter)*(z-z.mean(dim=0, keepdim=True)) #Original
        # z_pred = z_pred.reshape(hidden_states_kdims.shape)

        z = hidden_states_kdims.reshape(-1, hidden_states_kdims.shape[-1])
        z_pert = self.adapter_up(F.relu(self.adapter_down(z)))
        z_pred = z_pert

        hidden_states_dct_clone = hidden_states_dct.clone()
        hidden_states_dct_clone[:, :, :self.num_components] = z_pred
        reconstructed_hidden_states = self.idct1(hidden_states_dct_clone) + hidden_states

        # hidden_states_dct[:,:,:self.num_components] = z_pred
        # reconstructed_hidden_states = self.idct1(hidden_states_dct) #hidden_states
        # z_pert_out = self.whiten_and_recolor(z_pert,z)
        # print('reconstructed: ',reconstructed_hidden_states.shape)
        return reconstructed_hidden_states
    

# class MyCustomAdapter(nn.Module):
#     def __init__(self, hidden_dim=768, adapter_dim=16):
#         super().__init__()
#         self.adapter_down = nn.Linear(hidden_dim, adapter_dim)
#         self.adapter_up = nn.Linear(adapter_dim, hidden_dim)
#         self.adapter_act = nn.ReLU()

#     def forward(self, x):
#         return x + self.adapter_up(self.adapter_act(self.adapter_down(x)))