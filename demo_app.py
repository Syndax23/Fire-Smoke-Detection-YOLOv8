import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Yangın ve Duman Algılama Sistemi")


@st.cache_resource
def load_demo_model():
    return YOLO(r"C:\Users\ASUS\OneDrive\Desktop\FireDetectionProjesi\runs\detect\train\weights\best.pt")

model = load_demo_model()

uploaded_file = st.file_uploader("Test etmek için bir resim yükleyiniz...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yüklenen Orijinal Resim")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Model Çıkarım Sonucu")
        
        img_array = np.array(image)
        results = model(img_array)
        
        annotated_img = results[0].plot()
        
        st.image(annotated_img, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Sistem Tespit Raporu")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.error(f"Sistem Uyarı Durumu: {len(boxes)} adet nesne/belirti tespit edildi!")
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"• **Algılanan:** {label.upper()} | **Güven Oranı:** %{confidence * 100:.2f}")
    else:
        st.success("Sistem Durumu: Temiz. Herhangi bir tehlike algılanmadı.")