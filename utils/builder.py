from dataset import Grasp6DDataset_Train
from models import *
from utils.config_utils import simple_weights_init
from torch.utils.data import DataLoader
from torch.optim import Adam


model_pool = {
    "Denoiser": Denoiser
}

optimizer_pool = {
    "adam": Adam
}

init_pool = {
    "simple_weights_init": simple_weights_init
}


def build_dataset(cfg):
    """
    Function to build the dataset.
    """
    if hasattr(cfg, "data"):
        data_info = cfg.data
        dataset_path = data_info.dataset_path     # get the path to the dataset
        num_neg_prompts = data_info.num_neg_prompts   # get the maximum number of negative prompts
        train_set = Grasp6DDataset_Train(dataset_path, num_neg_prompts=num_neg_prompts) # the training set
        dataset_dict = dict(
            train_set=train_set,
        )
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


def build_loader(cfg, dataset_dict):
    """
    Function to build the loader
    """
    train_set = dataset_dict["train_set"]
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size, shuffle=True, drop_last=False, num_workers=8)
    loader_dict = dict(
        train_loader=train_loader,
    )

    return loader_dict


def build_model(cfg):
    """
    Function to build the model.
    """
    if hasattr(cfg, "model"):
        model_info = cfg.model
        weights_init = model_info.get("weights_init", None)
        model_name = model_info.type
        model_cls = model_pool[model_name]
        
        if model_name in ["Denoiser"]:
            betas = model_info.get("betas")
            n_T = model_info.get("n_T")
            drop_prob = model_info.get("drop_prob") 
            model = model_cls(n_T, betas, drop_prob)
        else:
            raise ValueError("Name of model does not exist!")
        if weights_init is not None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_optimizer(cfg, model):
    """
    Function to build the optimizer.
    """
    if hasattr(cfg, "optimizer"):
        optimizer_info = cfg.optimizer
        optimizer_type = optimizer_info.type
        optimizer_info.pop("type")
        optimizer_cls = optimizer_pool[optimizer_type]
        optimizer = optimizer_cls(model.parameters(), **optimizer_info)
        optim_dict = dict(
            optimizer=optimizer
        )
        return optim_dict
    else:
        raise ValueError("Configuration does not have optimizer config!")
