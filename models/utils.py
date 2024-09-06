import math
import torch
import torch.nn as nn
import open_clip
import torch.nn.functional as F
from .pointnet_utils import PointNetSetAbstractionMsg


def linear_diffusion_schedule(betas, n_T):
    """
    Linear scheduler for sampling in training.
    """
    beta_t = (betas[1] - betas[0]) * torch.arange(0, n_T + 1, dtype=torch.float64) / n_T + betas[0]
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
    

def cosine_diffusion_schedule(cosine_s, n_T):
    """
    Cosine scheduling for sampling in training.
    """
    timesteps = (
            torch.arange(n_T + 1, dtype=torch.float64) / n_T + cosine_s
        )
    alphas = timesteps / (1 + cosine_s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    beta_t = 1 - alphas[1:] / alphas[:-1]
    beta_t = beta_t.clamp(max=0.999)
    
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
    

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding for time step.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def __len__(self):
        return self.dim


class TextEncoder(nn.Module):
    """
    Text Encoder to encode the text prompt.
    """
    def __init__(self, device):
        super(TextEncoder, self).__init__()
        self.device = device
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k",
                                                                      device=self.device)
    
    def forward(self, texts):
        """
        texts can be a single string or a list of strings.
        """
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(texts).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device)
        return text_features
    

class PosNegTextEncoder(nn.Module):
    """
    Text encoder that performs differently for positive and negative prompts.
    """
    def __init__(self, device):
        super(PosNegTextEncoder, self).__init__()
        self.text_encoder = TextEncoder(device=device)
        
    def forward(self, texts, type):
        if type == "pos":   # if positive prompt
            return self.text_encoder(texts)
        elif type == "neg": # if negative prompts
            B = len(texts[0])
            l = list(zip(*texts))
            l = [item for sublist in l for item in sublist]
            embeddings = self.text_encoder(l)   # (B x 4) x 512
            embeddings = embeddings.reshape(B, -1, embeddings.shape[-1])
            return embeddings
        
    
class SceneEncoderPointNetPlusPlus(nn.Module):
    """
    Scene encoder based on PointNet++, returns scene tokens.
    """
    def __init__(self, additional_channel):
        super(SceneEncoderPointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1, 0.2], [
                                            32, 64, 128], 3+additional_channel, [[16, 16, 32], [32, 32, 64], [32, 48, 64]])
        self.sa2 = PointNetSetAbstractionMsg(
            512, [0.2, 0.4], [64, 128], 64+64+32, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [128, 256], 128+128, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):
        """
        Return point cloud embedding.
        """
        # Set Abstraction layers
        xyz = xyz.contiguous()
        
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points


class GraspNet(nn.Module):
    """
    Class to encoder the grasping pose.
    """
    def __init__(self):
        super(GraspNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
    
    def forward(self, g):
        return self.net(g)


class BNMomentum(object):
    """
    Class for BatchNormMomentum.
    """
    def __init__(self, origin_m, m_decay, step):
        super().__init__()
        self.origin_m = origin_m
        self.m_decay = m_decay
        self.step = step
        return

    def __call__(self, m, epoch):
        momentum = self.origin_m * (self.m_decay**(epoch//self.step))
        if momentum < 0.01:
            momentum = 0.01
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
        return