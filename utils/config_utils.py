import numpy as np
import torch
import random


class IOStream():
    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text+"\n")
        self.f.flush()

    def close(self):
        self.f.close()


def set_random_seed(seed):
    """
    Function to set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def simple_weights_init(m):
    """
    Function to initialize weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.state_dict().get("bias") is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.state_dict().get("bias") is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)