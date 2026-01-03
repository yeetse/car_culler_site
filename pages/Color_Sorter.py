import streamlit as st
from utils.inference import run_sorter

st.title("🎨 Sort by Color")

run_sorter(
    title="Upload car photos",
    model_path="resnet50_car_color_classifier_model.pt",
    mode="color"
)

st.markdown("---")

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.85rem;'>"
    "⚠️ Experimental feature — color predictions are still being improved."
    "</p>",
    unsafe_allow_html=True
)