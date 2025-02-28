import scanpy as sc
import torch

from perttf.model import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function import eval_testdata, wrapper_train

# Example structure
adata = None
h5ad_file = "/Users/zach/Documents/School/wei/pegasus/data/D18_diabetes_merged.h5ad"
adata = sc.read_h5ad(h5ad_file)
config_params = {
    "n_pert": 10,  # Number of perturbation types
    "nlayers_pert": 3,  # Perturbation decoder layers
    "n_hvg": 2000,  # Number of highly variable genes
    "mask_ratio": 0.15,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "perturbation_classifier_weight": 1.0,
    "cell_type_classifier_weight": 0.5,
    "pad_token": "<pad>",
    "pad_value": 0,
    "seed": 42,
    "ADV": False,
    "dab_weight": 1.0,
    "per_seq_batch_sample": False,
}
# Initialize config
config, run = generate_config(config_params, wandb_mode="online")

# Prepare data
data_dict = produce_training_datasets(
    adata,
    config,
    input_layer_key="X_binned",
    next_layer_key="X_binned_next",
    next_cell_pred="pert",
)

adata_sorted = adata[~adata.obs["is_in_training"]].copy()
test_adata = adata_sorted

# Initialize model
model = PerturbationTFModel(
    n_pert=config.n_pert,
    nlayers_pert=config.nlayers_pert,
    d_model=256,
    nhead=8,
    nlayers=6,
    n_cls=data_dict["n_cls"],
)

# Train
trained_model = wrapper_train(
    model, config, data_dict, save_dir="./saved_models", device="cuda"
)

# Load saved model
model.load_state_dict(torch.load("./saved_models/best_model.pt"))

# Generate predictions
results = eval_testdata(
    model,
    test_adata,
    gene_ids=data_dict["gene_ids"],
    train_data_dict=data_dict,
    config=config,
    include_types=["pert", "cls"],
)

# Access outputs:
pert_predictions = test_adata.obs["predicted_genotype"]
cell_embeddings = test_adata.obsm["X_scGPT"]
pert_probs = test_adata.obsm["X_pert_pred_probs"]
