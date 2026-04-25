import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset_sanskrit_images import SanskritImageDataset
from model_vfn import VisualFingerprintNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 2  


def make_dataloaders():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    ds_train = SanskritImageDataset(
        split="train",
        restrict_to_hf=True
    )
    ds_train.transform = transform

    ds_val = SanskritImageDataset(
        split="validation",
        restrict_to_hf=True
    )
    ds_val.transform = transform

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    return dl_train, dl_val


def train():
    dl_train, dl_val = make_dataloaders()

    model = VisualFingerprintNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in dl_train:
            imgs = imgs.to(DEVICE)

            
            y_mat = torch.tensor(labels["material_type"], dtype=torch.long).to(DEVICE)
            y_scr = torch.tensor(labels["script_family"], dtype=torch.long).to(DEVICE)
            y_man = torch.tensor(labels["is_manuscript"], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = (
                loss_fn(outputs["material_type"], y_mat)
                + loss_fn(outputs["script_family"], y_scr)
                + loss_fn(outputs["is_manuscript"], y_man)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dl_train))
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "vfn_model.pt")
    print("Saved model → vfn_model.pt")


if __name__ == "__main__":
    train()
