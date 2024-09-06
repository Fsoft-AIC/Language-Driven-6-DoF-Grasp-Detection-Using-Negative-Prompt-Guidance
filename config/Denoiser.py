import os


exp_name = "Denoiser"
seed = 1
log_dir = os.path.join("./log/", exp_name)
try:
    os.makedirs(log_dir)
except:
    print("Logging Dir is already existed!")

optimizer = dict(
    type="adam",
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4,
)

model = dict(
    type="Denoiser",
    betas=[1e-4, 2e-2],
    n_T=200,
    drop_prob=0.1,
)

training_cfg = dict(
    model=model,
    batch_size=128,
    epoch=200,
    gamma=0.9,  # used by the loss function
    workflow=dict(
        train=1,
    ),
)

data = dict(
    dataset_path="/cm/shared/toannt28/grasp-anything",
    num_neg_prompts=4
)