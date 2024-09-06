import os
import torch
from tqdm import tqdm
from utils import *
from models.utils import PosNegTextEncoder
from models.loss import DenoiserEuclideanLoss


class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.logger = running["logger"]
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.optimizer_dict = running["optim_dict"]
        self.optimizer = self.optimizer_dict.get("optimizer", None)
        self.epoch = 0
        
        self.gamma = cfg.training_cfg.get("gamma", 0.9) # gamma for loss functions
            
        # define text encoder
        self.posneg_text_encoder = PosNegTextEncoder(device=torch.device("cuda"))

    def train(self):
        denoiser_euclidean_loss = DenoiserEuclideanLoss()
        self.model.train()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        pbar = tqdm(self.train_loader)
        if self.epoch > 100:    # freeze the scene encoder aftr 100 epochs to accelerate the training.
            for p in self.model.noise_predictor.scene_encoder.parameters():
                p.requires_grad = False
            
        for _, pc, pos_prompt, neg_prompts, Rt, w in pbar:
            B = pc.shape[0]
            pc = pc.float().to("cuda")
            Rt, w = Rt.float().to("cuda"), w.float().to("cuda")
            noise = torch.randn(B, 7).to("cuda")
            with torch.no_grad():
                pos_prompt_embedding = self.posneg_text_encoder(pos_prompt, type="pos")
                neg_prompt_embeddings = self.posneg_text_encoder(neg_prompts, type="neg")
            predicted_noise, neg_prompt_pred, neg_prompt_embeddings = self.model(Rt, w, pc, pos_prompt_embedding, neg_prompt_embeddings, noise)
            mse_loss, neg_loss = denoiser_euclidean_loss(predicted_noise, noise, neg_prompt_pred, neg_prompt_embeddings)
            pbar.set_description(f"MSE loss: {mse_loss.item():.5f}, Neg loss: {neg_loss.item():.5f}")
            
            loss = self.gamma * mse_loss + (1 - self.gamma) * neg_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.logger.cprint(f"\nEpoch {self.epoch}, Real-time mse loss: {mse_loss.item():.5f},\
            Real-time neg loss: {neg_loss.item():.5f}")
        print("Saving checkpoint\n----------------------------------------\n")
        torch.save(self.model.state_dict(), os.path.join(self.cfg.log_dir, "current_model.t7"))
        self.epoch += 1
        
    def val(self):
       raise NotImplementedError

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        while self.epoch < EPOCH:
            for key, running_epoch in workflow.items():
                epoch_runner = getattr(self, key)
                for _ in range(running_epoch):
                    epoch_runner()
