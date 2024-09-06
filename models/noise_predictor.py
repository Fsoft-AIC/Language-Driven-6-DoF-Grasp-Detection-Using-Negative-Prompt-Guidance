import torch
import torch.nn as nn
from models.utils import SceneEncoderPointNetPlusPlus, GraspNet, SinusoidalPositionEmbeddings

            
class NoisePredictor(nn.Module):
    """
    This model uses PointNet++.
    This model uses text mask, which is the mask to drop the text guidance.
    This model also predicts the negative prompt guidance.
    This model is for the Euclidean distance-based loss function for the learning of negative prompt guidance.
    """
    def __init__(self, time_emb_dim):
        """
        This model uses PointNet++.
        time_emd_dim: dimension of the point embedding.
        """
        super(NoisePredictor, self).__init__()
        self.scene_encoder = SceneEncoderPointNetPlusPlus(additional_channel=3)
        self.time_encoder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.grasp_encoder = GraspNet()
        self.negative_net = nn.Sequential(  # this module output the predicted negative embedding
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        self.grasp_pos_prompt_time_net = nn.Sequential( # this module is to embed the concatenation of grasp + pos_prompt's, and time's embeddings.
            nn.Linear(512 + time_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.final_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 7)   # 7 = 6 (for Rt) + 1 (for w)
        )
        
    def forward(self, g, pc, pos_prompt_embedding, neg_prompt_embeddings, text_mask, time):
        """
        pc's size is B x N x D
        text_mask's size is B x 1. The text_mask is used for positive prompt only. No mask for the pc.
        time's size is B
        g is in se(3)
        """
        pc = pc.permute(0, 2, 1)
        scene_tokens = self.scene_encoder(pc)   # B x 512 x 128, for the scenes of 8192 points
        scene_tokens = scene_tokens.permute(0, 2, 1)    # B x 128 x 512
        scene_embedding = torch.mean(scene_tokens, dim=1)   # B x 512, the embeddings for entire scene
        
        neg_prompt_pred = self.negative_net(scene_embedding - pos_prompt_embedding) # predict the negative prompt
        masked_pos_prompt_embedding = pos_prompt_embedding * text_mask  # B x 512, drop the positive prompt using the text_mask
        
        grasp_embedding = self.grasp_encoder(g)  # B x 512
        grasp_pos_prompt_embedding = grasp_embedding + masked_pos_prompt_embedding  # B x 512
        time_embedding = self.time_encoder(time)    # B x 64, get the time positional embedding
        grasp_pos_prompt_time_embedding = self.grasp_pos_prompt_time_net(torch.cat((grasp_pos_prompt_embedding, time_embedding), dim=1)).unsqueeze(1)   # B x 1 x 512
        
        e, _ = self.cross_attention(query=grasp_pos_prompt_time_embedding, key=scene_tokens, value=scene_tokens)  # B x 1 x 512
        predicted_noise = self.final_net(e.squeeze(1) + grasp_pos_prompt_time_embedding.squeeze(1))   # residual connection
        
        return predicted_noise, neg_prompt_pred, neg_prompt_embeddings
    
    def forward_precomputing(self, g, scene_tokens, pos_prompt_embedding, text_mask, time):
        """
        This performs given pre-computed scene_tokens.
        The neg_prompt_pred is not used.
        """
        masked_pos_prompt_embedding = pos_prompt_embedding * text_mask  # B x 512, drop the positive prompt using the text_mask
        
        grasp_embedding = self.grasp_encoder(g)  # B x 512
        grasp_pos_prompt_embedding = grasp_embedding + masked_pos_prompt_embedding  # B x 512
        time_embedding = self.time_encoder(time)    # B x 64, get the time positional embedding
        grasp_pos_prompt_time_embedding = self.grasp_pos_prompt_time_net(torch.cat((grasp_pos_prompt_embedding, time_embedding), dim=1)).unsqueeze(1)   # B x 1 x 512
        
        e, _ = self.cross_attention(query=grasp_pos_prompt_time_embedding, key=scene_tokens, value=scene_tokens)  # B x 1 x 512
        predicted_noise = self.final_net(e.squeeze(1) + grasp_pos_prompt_time_embedding.squeeze(1))   # residual connection
        
        return predicted_noise
