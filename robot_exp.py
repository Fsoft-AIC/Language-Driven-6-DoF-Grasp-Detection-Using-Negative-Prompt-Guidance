import os
import torch
import pickle
from gorilla.config import Config
from utils import *
import argparse
from models.utils import PosNegTextEncoder
from pytorchse3.se3 import se3_exp_map

# LOAD YOUR POINT CLOUD HERE, make sure its size is N x (3+3), 3 for coordinate and 3 for color,
# N is the number of points, can be varied, but preferred to be 8192
pc = None
# SPECIFY YOUR TEXT, for example, "Grasp me the pencil."
text = None

def parse_args():
    parser = argparse.ArgumentParser(description="Robot experiments")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint model")
    parser.add_argument("--n_sample", type=int, help="number of samples to generate for the\
                        point cloud-text pair")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get("seed") is not None:
        set_random_seed(cfg.seed)
    
    # Build the model
    model = build_model(cfg)
    model = model.to("cuda")
    model = torch.nn.DataParallel(model)    
    model.load_state_dict(torch.load(args.checkpoint))
    posneg_text_encoder = PosNegTextEncoder(device=torch.device("cuda"))
    n_sample = args.n_sample
    
    model.eval()
    
    pc = pc.unsqueeze(0).repeat(n_sample, 1, 1).float().to("cuda")
    with torch.no_grad():
        text_embedding = posneg_text_encoder(text, type="pos").repeat(n_sample, 1)
    generated_grasps = se3_exp_map(model.module.generate(pc, text_embedding)).cpu().detach().numpy() # use 1 GPU only
    
    # Save generated grasps to file
    with open(os.path.join(cfg.log_dir, "generated_grasps.pkl"), 'wb') as f:
        pickle.dump(generated_grasps, f) 
