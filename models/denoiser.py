import torch
import torch.nn as nn
from pytorchse3.se3 import se3_log_map as pytorchse3_log_map
from utils import *
from models.noise_predictor import NoisePredictor
from models.utils import linear_diffusion_schedule


class Denoiser(nn.Module):
    """
    This model is for the Euclidean-based distance.
    """
    def __init__(self, n_T, betas, drop_prob=0.1):   # probability to drop the text
        super(Denoiser, self).__init__()
        self.noise_predictor = NoisePredictor(time_emb_dim=64)
        for k, v in linear_diffusion_schedule(betas, n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.drop_prob = drop_prob
    
    def forward(self, Rt, w, pc, pos_prompt_embedding, neg_prompt_embeddings, noise):
        B = pc.shape[0]
        time = torch.randint(1, self.n_T + 1, (B,)).to("cuda")
        g = torch.cat((pytorchse3_log_map(Rt), w[:, None]), dim=-1)
        g_t = (self.sqrtab[time, None] * g + self.sqrtmab[time, None] * noise).float()
        text_mask = torch.bernoulli(torch.zeros(B, 1) + 1 - self.drop_prob).to("cuda")
        predicted_noise, neg_prompt_pred, neg_prompt_embedding = self.noise_predictor(g_t, pc, pos_prompt_embedding, neg_prompt_embeddings, text_mask, time)
        
        return predicted_noise, neg_prompt_pred, neg_prompt_embedding
    
    def generate(self, pc, pos_prompt_embedding, w=1.0):
        """"
        pc's size: n_sample x 8192 x 6.
        pos_prompt_embedding's size: n_sample x 512
        """
        # Pre-compute the scene tokens
        pc = pc.permute(0, 2, 1)    # B x D x N
        scene_tokens = self.noise_predictor.scene_encoder(pc).permute(0, 2, 1)
        
        # Pre-compute the the negative prompt guidance
        scene_embedding = torch.mean(scene_tokens, dim=1)   # B x 512, the embeddings for entire scene
        neg_prompt_embedding = self.noise_predictor.negative_net(scene_embedding - pos_prompt_embedding)
        text_embedding = torch.cat((pos_prompt_embedding.repeat(2, 1), neg_prompt_embedding), axis=0)
        scene_tokens = scene_tokens.repeat(3, 1, 1)
        
        n_sample = pc.shape[0]
        g_i = torch.randn(n_sample, (7)).to("cuda")
        text_mask = torch.ones_like(pos_prompt_embedding).to("cuda")
        text_mask = text_mask.repeat(3, 1)
        text_mask[:n_sample] = 0.   # make the first part text-free
        for j in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, (7)) if j > 1 else torch.zeros((n_sample, 7)).float()
            z = z.to("cuda")
            
            g_i = g_i.repeat(3, 1) 
            time = torch.tensor([j]).repeat(3*n_sample).to("cuda")
            eps = self.noise_predictor.forward_precomputing(g_i, scene_tokens, text_embedding, text_mask, time)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:2*n_sample]
            eps3 = eps[2*n_sample:]
            eps = eps1 + w * (eps2 - eps3)
            eps = torch.clamp(eps, -1.0, 1.0)
            g_i = g_i[:n_sample]
            g_i = self.oneover_sqrta[j] * (g_i - eps * self.mab_over_sqrtmab[j]) + self.sqrt_beta_t[j] * z
            
        return g_i