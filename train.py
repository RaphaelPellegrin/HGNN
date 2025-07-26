import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
from datasets.visual_data import load_explicit_hypergraph

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cfg = get_config("config/config.yaml")

# initialize data
if cfg["on_dataset"] == "ModelNet40":
    data_dir = cfg["modelnet40_ft"]
elif cfg["on_dataset"] == "NTU2012":
    data_dir = cfg["ntu2012_ft"]
elif cfg["on_dataset"] == "CoauthorshipCora":
    data_dir = cfg["coauthorship_cora_ft"]
elif cfg["on_dataset"] == "CoauthorshipDblp":
    data_dir = cfg["coauthorship_dblp_ft"]
elif cfg["on_dataset"] == "CocitationCora":
    data_dir = cfg["cocitation_cora_ft"]
elif cfg["on_dataset"] == "CocitationCiteseer":
    data_dir = cfg["cocitation_citeseer_ft"]
elif cfg["on_dataset"] == "CocitationPubmed":
    data_dir = cfg["cocitation_pubmed_ft"]
else:
    raise ValueError(f"Unknown dataset: {cfg['on_dataset']}")

# Check if we should use explicit hypergraph structure
if cfg.get("use_explicit_hypergraph", False):
    print("🕸️ Using explicit hypergraph structure!")
    result = load_explicit_hypergraph(data_dir)
    if result is not None:
        fts, lbls, idx_train, idx_test, H = result
        print(f"✅ Using explicit hypergraph: {H.shape}")
    else:
        print("⚠️ Falling back to feature-based hypergraph construction")
        fts, lbls, idx_train, idx_test, H = load_feature_construct_H(
            data_dir,
            m_prob=cfg["m_prob"],
            K_neigs=cfg["K_neigs"],
            is_probH=cfg["is_probH"],
            use_mvcnn_feature=cfg["use_mvcnn_feature"],
            use_gvcnn_feature=cfg["use_gvcnn_feature"],
            use_mvcnn_feature_for_structure=cfg["use_mvcnn_feature_for_structure"],
            use_gvcnn_feature_for_structure=cfg["use_gvcnn_feature_for_structure"],
        )
else:
    print("🔨 Using feature-based hypergraph construction (KNN)")
    fts, lbls, idx_train, idx_test, H = load_feature_construct_H(
        data_dir,
        m_prob=cfg["m_prob"],
        K_neigs=cfg["K_neigs"],
        is_probH=cfg["is_probH"],
        use_mvcnn_feature=cfg["use_mvcnn_feature"],
        use_gvcnn_feature=cfg["use_gvcnn_feature"],
        use_mvcnn_feature_for_structure=cfg["use_mvcnn_feature_for_structure"],
        use_gvcnn_feature_for_structure=cfg["use_gvcnn_feature_for_structure"],
    )
G = hgut.generate_G_from_H(H)
n_class = int(lbls.max()) + 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(
    model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, patience=50
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print("-" * 10)
            print(f"Epoch {epoch}/{num_epochs - 1}")

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == "train" else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0  # Reset counter
                print(f"   🎯 New best validation accuracy: {best_acc:.4f}")
            elif phase == "val":
                epochs_without_improvement += 1

        if epoch % print_freq == 0:
            print(f"Best val Acc: {best_acc:4f}")
            print(
                f"Epochs without improvement: {epochs_without_improvement}/{patience}"
            )
            print("-" * 20)

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs!")
            print(f"No improvement in validation accuracy for {patience} epochs.")
            early_stopped = True
            break

    time_elapsed = time.time() - since
    if early_stopped:
        print(
            f"\n⏰ Training stopped early in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
    else:
        print(
            f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
    print(f"Best val Acc: {best_acc:4f}")
    print(f"Final epochs without improvement: {epochs_without_improvement}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print("Configuration -> Start")
    pp.pprint(cfg)
    print("Configuration -> End")

    model_ft = HGNN(
        in_ch=fts.shape[1], n_class=n_class, n_hid=cfg["n_hid"], dropout=cfg["drop_out"]
    )
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(
        model_ft.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    model_ft = train_model(
        model_ft,
        criterion,
        optimizer,
        schedular,
        cfg["max_epoch"],
        print_freq=cfg["print_freq"],
        patience=cfg.get("patience", 50),  # Default to 50 if not specified
    )


if __name__ == "__main__":
    _main()
