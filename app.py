import streamlit as st
import torch
import zipfile
from pathlib import Path
from torchvision import models, transforms
from utils import sort_uploaded_images

# Config
st.set_page_config(page_title="Car Photo Sorter", layout="centered")
st.title("🚗 Car Photo Color Sorter")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model (cached)
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 8)  # adjust to your number of classes
    )
    model = torch.load("/Users/Yeeva/IdeaProjects/car_culler_site/resnet50_car_color_classifier_model.pt", weights_only=False, map_location=device)
    model.to(device)
    model.eval()
    return model

model = load_model()
class_names = ['Black', 'Blue', 'Brown', 'Cyan', 'Green', 'Grey', 'Orange', 'Red',
               'Violet',
               'White',
               'Yellow']

# Transforms
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Upload
uploaded_files = st.file_uploader(
    "Upload car photos",
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} images uploaded")

    if st.button("Sort Photos"):
        output_dir = Path("sorted_output")
        sort_uploaded_images(
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
