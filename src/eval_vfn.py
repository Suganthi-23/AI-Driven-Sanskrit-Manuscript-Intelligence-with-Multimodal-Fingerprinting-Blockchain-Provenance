from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from dataset_sanskrit_images import (
    SanskritImageDataset,
    MATERIAL_CLASSES,
    SCRIPT_CLASSES,
    MANUSCRIPT_FLAGS,
)
from model_vfn import VisualFingerprintNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_PATH = Path("vfn_model.pt")


def make_val_loader():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    ds_val = SanskritImageDataset(
        split="validation",
        restrict_to_hf=True
    )
    ds_val.transform = transform

    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    return dl_val


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    print(f"[*] Loading model from {MODEL_PATH}")
    model = VisualFingerprintNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    dl_val = make_val_loader()

    all_true_mat, all_pred_mat = [], []
    all_true_scr, all_pred_scr = [], []
    all_true_man, all_pred_man = [], []

    with torch.no_grad():
        for imgs, labels in dl_val:
            imgs = imgs.to(DEVICE)

            y_mat = labels["material_type"]
            y_scr = labels["script_family"]
            y_man = labels["is_manuscript"]

            outputs = model(imgs)

            pred_mat = outputs["material_type"].argmax(dim=1).cpu()
            pred_scr = outputs["script_family"].argmax(dim=1).cpu()
            pred_man = outputs["is_manuscript"].argmax(dim=1).cpu()

            all_true_mat.extend(y_mat)
            all_pred_mat.extend(pred_mat)
            all_true_scr.extend(y_scr)
            all_pred_scr.extend(pred_scr)
            all_true_man.extend(y_man)
            all_pred_man.extend(pred_man)

    # Convert to int lists
    all_true_mat = [int(x) for x in all_true_mat]
    all_pred_mat = [int(x) for x in all_pred_mat]
    all_true_scr = [int(x) for x in all_true_scr]
    all_pred_scr = [int(x) for x in all_pred_scr]
    all_true_man = [int(x) for x in all_true_man]
    all_pred_man = [int(x) for x in all_pred_man]

    # Dynamic labels
    labels_mat = unique_labels(all_true_mat)
    labels_scr = unique_labels(all_true_scr)
    labels_man = unique_labels(all_true_man)

    print("\n=== MATERIAL TYPE REPORT ===")
    print(classification_report(
        all_true_mat, all_pred_mat,
        labels=labels_mat,
        target_names=[MATERIAL_CLASSES[i] for i in labels_mat]
    ))
    print("Confusion matrix:\n", confusion_matrix(all_true_mat, all_pred_mat, labels=labels_mat))

    print("\n=== SCRIPT FAMILY REPORT ===")
    print(classification_report(
        all_true_scr, all_pred_scr,
        labels=labels_scr,
        target_names=[SCRIPT_CLASSES[i] for i in labels_scr]
    ))
    print("Confusion matrix:\n", confusion_matrix(all_true_scr, all_pred_scr, labels=labels_scr))

    print("\n=== IS_MANUSCRIPT REPORT ===")
    print(classification_report(
        all_true_man, all_pred_man,
        labels=labels_man,
        target_names=[MANUSCRIPT_FLAGS[i] for i in labels_man]
    ))
    print("Confusion matrix:\n", confusion_matrix(all_true_man, all_pred_man, labels=labels_man))


if __name__ == "__main__":
    evaluate()
