import torch
import shutil
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

VALID_EXTS = {".jpg", ".jpeg", ".png"}

class UploadDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.files[idx].name


def sort_uploaded_images(
        model,
        uploaded_files,
        transform,
        class_names,
        output_dir,
        device
):
    output_dir.mkdir(exist_ok=True)

    ds = UploadDataset(uploaded_files, transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model.eval()
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            for name, pred in zip(names, preds):
                cls = class_names[pred]
                cls_dir = output_dir / cls
                cls_dir.mkdir(exist_ok=True)

                for f in uploaded_files:
                    if f.name == name:
                        with open(cls_dir / name, "wb") as out:
                            out.write(f.getbuffer())
