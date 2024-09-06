import os
import torch
import pickle
from gorilla.config import Config
from utils import *
import argparse
from tqdm import tqdm
from models.utils import PosNegTextEncoder
from dataset import Grasp6DDataset_Test


def parse_args():
    parser = argparse.ArgumentParser(description="Test a model by generating grasps")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint model")
    parser.add_argument("--data_path", type=str, help="path to test dataset")
    parser.add_argument("--n_sample", type=int, help="number of samples to generate for a\
                        point cloud-text pair")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get("seed") is not None:
        set_random_seed(cfg.seed)
    model = build_model(cfg)
    model = model.to("cuda")
    model = torch.nn.DataParallel(model)
    
    # Load test data
    test_data = Grasp6DDataset_Test(args.data_path).all_data
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint))
    posneg_text_encoder = PosNegTextEncoder(device=torch.device("cuda"))
    n_sample = args.n_sample
    
    print("Generating...")
    model.eval()
    for datapoint in tqdm(test_data):
        """
        Each datapoint includes a point cloud, a positive prompt,
        and a set of corresponding grasp poses.
        """
        pc = torch.from_numpy(datapoint["pc"])
        pos_prompt = datapoint["pos_prompt"]
        pc = pc.unsqueeze(0).repeat(n_sample, 1, 1).float().to("cuda")
        with torch.no_grad():
            pos_prompt_embedding = posneg_text_encoder(pos_prompt, type="pos").repeat(n_sample, 1)
        generated_grasps = model.module.generate(pc, pos_prompt_embedding, w=0.2).cpu().detach().numpy() # use 1 GPU only
        datapoint["gen_grasps"] = generated_grasps
    
    with open(os.path.join(cfg.log_dir, "all_data.pkl"), 'wb') as f:
        pickle.dump(test_data, f) 
