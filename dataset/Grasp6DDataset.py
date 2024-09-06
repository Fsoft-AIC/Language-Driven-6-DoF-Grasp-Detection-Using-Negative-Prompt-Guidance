import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from pytorchse3.se3 import se3_log_map

MAX_WIDTH = 0.202   # maximum width of gripper 2F-140


class Grasp6DDataset_Train(Dataset):
    """
    Data loading class for training.
    """
    def __init__(self, dataset_path: str, num_neg_prompts=4):
        """
        dataset_path (str): path to the dataset
        num_neg_prompts: number of negative prompts used in training
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.num_neg_prompts = num_neg_prompts
        self._load()

    def _load(self):
        self.all_data = []
        filenames = sorted(os.listdir(f"{self.dataset_path}/pc"))
        
        print("Processing dataset for training!")
        filenames = filenames[:int(len(filenames)*4/5)]   # 80% scenes for training
            
        for filename in filenames:
            scene, _ = os.path.splitext(filename)
            pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
            try: 
                with open(f"{self.dataset_path}/grasp_prompt/{scene}.pkl", "rb") as f:
                    prompts = pickle.load(f)
            except:
                continue
            num_objects = len(prompts)
            for i in range(num_objects):
                try:
                    with open(f"{self.dataset_path}/grasp/{scene}_{i}.pkl", "rb") as f:
                        Rts, ws = pickle.load(f)
                except:
                    continue
                pos_prompt = prompts[i] # positive prompt
                neg_prompts = prompts[:i] + prompts[i + 1:]    # negative prompts
                real_num_neg_prompts = len(neg_prompts)
                if 0 < real_num_neg_prompts < self.num_neg_prompts:
                    neg_prompts = neg_prompts + [neg_prompts[-1]] * (self.num_neg_prompts - real_num_neg_prompts)    # pad with last negative prompt
                elif real_num_neg_prompts == 0: # if no negative prompt
                    neg_prompts = [""] * self.num_neg_prompts   # then use empty strings
                else:   # if the real number of negative prompts exceeeds self.num_neg_prompts
                    neg_text = neg_text[:self.num_neg_text]
                
                self.all_data.extend([{"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "neg_prompts": neg_prompts,\
                    "Rt": Rt, "w": 2*w/MAX_WIDTH-1.0} for Rt, w in zip(Rts, ws)])
        
        return self.all_data
            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        element = self.all_data[index]
        return element["scene"], element["pc"], element["pos_prompt"] , element["neg_prompts"], element["Rt"], element["w"]      

    def __len__(self):
        return len(self.all_data)
    
    
class Grasp6DDataset_Test(Dataset):
    """
    Data loading class for testing.
    """
    def __init__(self, dataset_path: str):
        """
        dataset_path (str): path to the dataset
        """
        super().__init__()
        self.dataset_path = dataset_path
        self._load()

    def _load(self):
        self.all_data = []
        filenames = sorted(os.listdir(f"{self.dataset_path}/pc"))
        
        print("Processing dataset for testing!")
        filenames = filenames[int(len(filenames)*4/5):]   # 20% scenes for testing
            
        for filename in filenames:
            scene, _ = os.path.splitext(filename)
            pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
            try: 
                with open(f"{self.dataset_path}/grasp_prompt/{scene}.pkl", "rb") as f:
                    prompts = pickle.load(f)
            except:
                continue
            num_objects = len(prompts)
            for i in range(num_objects):
                try:
                    with open(f"{self.dataset_path}/grasp/{scene}_{i}.pkl", "rb") as f:
                        Rts, ws = pickle.load(f)
                except:
                    continue
                pos_prompt = prompts[i] # positive prompt
                gs = np.concatenate((se3_log_map(torch.from_numpy(Rts)).numpy(), 2*ws[:, None]/MAX_WIDTH-1.0), axis=-1)
                self.all_data.append({"scene": scene, "pc": pc, "pos_prompt": pos_prompt, "gs": gs})
        
        return self.all_data
            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        element = self.all_data[index]
        return element["scene"], element["pc"], element["pos_prompt"] , element["gs"]      

    def __len__(self):
        return len(self.all_data)