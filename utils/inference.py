import torch
import shutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import zipfile
from pathlib import Path
from torchvision import models, transforms
import streamlit as st

VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UploadDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.files[idx].name

def sort_by_color(
        model,
        uploaded_files,
        transform,
        class_names,
        output_dir,
        device
):
    # Clear folder for each run
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

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

def run_sorter(title, model_path, mode):
    st.header(title)

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} images uploaded")
        if st.button("Sort Photos"):
            # load model based on mode
            # run inference
            # clear output dir
            # zip results
            if mode == "color":
                # Load Model (cached)
                @st.cache_resource
                def load_model():
                    loaded_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                    loaded_model.fc = torch.nn.Sequential(
                        torch.nn.Linear(loaded_model.fc.in_features, 512),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.4),
                        torch.nn.Linear(512, 11)
                    )
                    loaded_model = torch.load(model_path, weights_only=False, map_location=device)
                    loaded_model.to(device)
                    loaded_model.eval()
                    return loaded_model

                model = load_model()
                class_names = ['Black', 'Blue', 'Brown', 'Cyan', 'Green', 'Grey', 'Orange', 'Red', 'Violet', 'White', 'Yellow']

                # Transforms
                val_tfms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])

                output_dir = Path("sorted_output")
                sort_by_color(
                    model,
                    uploaded_files,
                    val_tfms,
                    class_names,
                    output_dir,
                    device
                )

                # Zip results
                zip_path = "sorted_photos.zip"
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for file in output_dir.rglob("*"):
                        zipf.write(file, file.relative_to(output_dir))

                st.success("Sorting complete!")

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "Download Sorted Photos",
                        f,
                        file_name="sorted_photos.zip"
                    )

