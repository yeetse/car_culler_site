import streamlit as st

st.set_page_config(page_title="Car Culler", layout="centered")

st.title("🚗 CarCullr")
st.subheader("Smart photo culling for car photographers")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎨 Sort by Color")
    st.write("Group photos by car color for clean Instagram sets.")
    st.page_link("pages/Color_Sorter.py", label="Open Color Sorter")

with col2:
    st.markdown("### 🚘 Sort by Brand (Coming Soon)")
    st.write("Group photos by manufacturer.")
    st.button("Coming Soon", disabled=True)