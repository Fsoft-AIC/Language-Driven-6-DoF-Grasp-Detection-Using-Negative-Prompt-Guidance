import pickle
import numpy as np
from utils import *
import argparse
from tqdm import tqdm
from utils.test_utils import earth_movers_distance, coverage_rate, collision_free_rate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--data", type=str, help="path to the data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.data, "rb") as f:
       generated_data = pickle.load(f)
    cvr = np.array([coverage_rate(datapoint["gs"], datapoint["gen_grasps"])\
                    for datapoint in tqdm(generated_data)\
                    if coverage_rate(datapoint["gs"], datapoint["gen_grasps"]) is not None])
    print(f"Average CR: {np.mean(cvr)}")
    emds = np.array([earth_movers_distance(datapoint["gs"], datapoint["gen_grasps"])\
                    for datapoint in tqdm(generated_data)\
                    if earth_movers_distance(datapoint["gs"], datapoint["gen_grasps"]) is not None])
    print(f"Average EMD: {np.mean(emds)}")
    cfr = np.array([collision_free_rate(datapoint["pc"], datapoint["gen_grasps"])\
                    for datapoint in tqdm(generated_data)\
                    if collision_free_rate(datapoint["pc"], datapoint["gen_grasps"]) is not None])
    print(f"Average CFR: {np.mean(cfr)}")
        
