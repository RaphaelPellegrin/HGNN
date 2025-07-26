import scipy.io as scio
import numpy as np


def load_ft(data_dir, feature_name="GVCNN"):
    data = scio.loadmat(data_dir)
    lbls = data["Y"].astype(np.int64)
    if lbls.min() == 1:
        lbls = lbls - 1

    # Handle different indices formats
    indices_data = data["indices"]
    if indices_data.size == 1:
        # Original format: nested structure
        idx = indices_data.item()
    else:
        # Converted format: flat array
        idx = indices_data.flatten()

    if feature_name == "MVCNN":
        if data["X"].shape == (2, 1):
            # Original format: (2, 1) shape with nested items
            fts = data["X"][0].item().astype(np.float32)
        else:
            # Converted format: (1, 2) shape with direct arrays
            fts = data["X"][0, 0].astype(np.float32)
    elif feature_name == "GVCNN":
        if data["X"].shape == (2, 1):
            # Original format: (2, 1) shape with nested items
            fts = data["X"][1].item().astype(np.float32)
        else:
            # Converted format: (1, 2) shape with direct arrays
            fts = data["X"][0, 1].astype(np.float32)
    else:
        print(f"wrong feature name{feature_name}!")
        raise IOError

    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test
