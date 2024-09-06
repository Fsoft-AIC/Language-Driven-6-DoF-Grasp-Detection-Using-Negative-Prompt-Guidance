import torch
import torch.nn as nn
import torch.nn.functional as F

    
class DenoiserEuclideanLoss(nn.Module):
    def __init__(self):
        super(DenoiserEuclideanLoss, self).__init__()

    def forward(self, predicted_noise, noise, neg_prompt_pred, neg_prompt_embeddings):
        """
        neg_prompt_pred's size is B x 512
        neg_prompt_embeddings' size is B x num_neg_prompts x 512
        """
        mse_loss = F.mse_loss(predicted_noise, noise)
        neg_prompt_pred = neg_prompt_pred.unsqueeze(1).expand_as(neg_prompt_embeddings) # B x num_neg_prompts x 512
        paired_distances = torch.sqrt(torch.sum((neg_prompt_pred - neg_prompt_embeddings)**2, dim=2))
        neg_loss = torch.mean(torch.min(paired_distances, dim=1)[0]) # use minimum

        return mse_loss, neg_loss