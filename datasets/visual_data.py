from datasets import load_ft
from utils import hypergraph_utils as hgut
import scipy.io as scio
import numpy as np


def load_explicit_hypergraph(data_dir):
    """
    Load explicit hypergraph structure from converted datasets.

    Args:
        data_dir: Path to the .mat file containing explicit hypergraph

    Returns:
        tuple: (features, labels, train_indices, test_indices, hypergraph_matrix)
    """
    print("🕸️ Loading explicit hypergraph structure...")

    # Load the dataset
    data = scio.loadmat(data_dir)

    # Check if explicit hypergraph exists
    if "explicit_hypergraph" not in data:
        print(
            "⚠️ No explicit hypergraph found! Falling back to feature-based construction."
        )
        return None

    # Load features and other data first to get dimensions
    fts, lbls, idx_train, idx_test = load_ft(data_dir, feature_name="GVCNN")
    n_nodes = fts.shape[0]

    print(f"   ✅ Loaded features: {fts.shape}")
    print(f"   ✅ Loaded labels: {lbls.shape}, classes: {len(np.unique(lbls))}")
    print(f"   ✅ Train/test split: {len(idx_train)}/{len(idx_test)}")

    # Load explicit hypergraph
    H_explicit = data["explicit_hypergraph"]
    if hasattr(H_explicit, "toarray"):
        # Convert sparse to dense if needed for compatibility
        H_explicit = H_explicit.toarray()

    print(f"   📊 Original explicit hypergraph: {H_explicit.shape}")
    print(
        f"   Density: {np.count_nonzero(H_explicit)}/{H_explicit.size} ({np.count_nonzero(H_explicit)/H_explicit.size*100:.4f}%)"
    )

    # Handle dimension mismatch between hypergraph and features
    if H_explicit.shape[0] != n_nodes:
        print(
            f"   ⚠️ Dimension mismatch: hypergraph nodes ({H_explicit.shape[0]}) != feature nodes ({n_nodes})"
        )

        if H_explicit.shape[0] < n_nodes:
            # Pad the hypergraph to match feature dimensions
            print(f"   🔧 Padding hypergraph to match feature dimensions...")
            # Create identity hypergraph for missing nodes
            missing_nodes = n_nodes - H_explicit.shape[0]
            identity_part = np.eye(missing_nodes, dtype=H_explicit.dtype)

            # Combine original hypergraph with identity for missing nodes
            # Top part: original hypergraph with zeros for new hyperedges
            top_part = np.hstack(
                [H_explicit, np.zeros((H_explicit.shape[0], missing_nodes))]
            )
            # Bottom part: zeros for original hyperedges, identity for new nodes
            bottom_part = np.hstack(
                [np.zeros((missing_nodes, H_explicit.shape[1])), identity_part]
            )

            H_explicit = np.vstack([top_part, bottom_part])

        else:
            # Truncate the hypergraph to match feature dimensions
            print(f"   🔧 Truncating hypergraph to match feature dimensions...")
            H_explicit = H_explicit[:n_nodes, :]

    print(f"   ✅ Final explicit hypergraph: {H_explicit.shape}")
    print(
        f"   Final density: {np.count_nonzero(H_explicit)}/{H_explicit.size} ({np.count_nonzero(H_explicit)/H_explicit.size*100:.4f}%)"
    )

    return fts, lbls, idx_train, idx_test, H_explicit


def load_feature_construct_H(
    data_dir,
    m_prob=1,
    K_neigs=[10],
    is_probH=True,
    split_diff_scale=False,
    use_mvcnn_feature=False,
    use_gvcnn_feature=True,
    use_mvcnn_feature_for_structure=False,
    use_gvcnn_feature_for_structure=True,
):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    # init feature
    if use_mvcnn_feature or use_mvcnn_feature_for_structure:
        mvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name="MVCNN")
    if use_gvcnn_feature or use_gvcnn_feature_for_structure:
        gvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name="GVCNN")
    if "mvcnn_ft" not in dir() and "gvcnn_ft" not in dir():
        raise Exception("None feature initialized")

    # construct feature matrix
    fts = None
    if use_mvcnn_feature:
        fts = hgut.feature_concat(fts, mvcnn_ft)
    if use_gvcnn_feature:
        fts = hgut.feature_concat(fts, gvcnn_ft)
    if fts is None:
        raise Exception(f"None feature used for model!")

    # construct hypergraph incidence matrix
    print(
        "Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)"
    )
    H = None
    if use_mvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(
            mvcnn_ft,
            K_neigs=K_neigs,
            split_diff_scale=split_diff_scale,
            is_probH=is_probH,
            m_prob=m_prob,
        )
        H = hgut.hyperedge_concat(H, tmp)
    if use_gvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(
            gvcnn_ft,
            K_neigs=K_neigs,
            split_diff_scale=split_diff_scale,
            is_probH=is_probH,
            m_prob=m_prob,
        )
        H = hgut.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception("None feature to construct hypergraph incidence matrix!")

    return fts, lbls, idx_train, idx_test, H
