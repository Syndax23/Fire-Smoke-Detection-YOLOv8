import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Yangın & Duman Algılama", page_icon="🔥", layout="wide")
st.title("🔥 Yangın ve Duman Algılama Sistemi")

# --- Model Yükleme ---
@st.cache_resource
def load_demo_model():
    return YOLO(r"C:\Users\ASUS\OneDrive\Desktop\FireDetectionProjesi\runs\detect\fire_smoke_v2\weights\best.pt")

model = load_demo_model()

# --- Sidebar: Ayarlar ---
st.sidebar.header("⚙️ Algılama Ayarları")
confidence_threshold = st.sidebar.slider(
    "Güven Eşiği (Confidence Threshold)",
    min_value=0.05,
    max_value=0.95,
    value=0.15,       # 0.25 yerine 0.15 — fire için daha hassas algılama
    step=0.05,
    help="Düşük değer = daha fazla tespit (yanlış pozitif olabilir). Yüksek değer = daha az ama emin tespitler."
)

iou_threshold = st.sidebar.slider(
    "IOU Eşiği",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
    help="Üst üste gelen kutucukların birleştirilmesi için eşik."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **İpucu:** Yangın tespit edilemiyorsa Güven Eşiğini 0.10'a veya 0.05 düşürün.")

# --- Görüntü Yükleme ---
uploaded_file = st.file_uploader(
    "Test etmek için bir resim yükleyiniz...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Yüklenen Orijinal Resim")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Model Çıkarım Sonucu")

        img_array = np.array(image)

        results = model(
            img_array,
            conf=confidence_threshold,
            iou=iou_threshold
        )

        annotated_img = results[0].plot()
        st.image(annotated_img, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sistem Tespit Raporu")

    boxes = results[0].boxes

    if len(boxes) > 0:
        # Tespit edilen sınıfları say
        fire_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == "fire")
        smoke_count = sum(1 for box in boxes if model.names[int(box.cls[0])] == "smoke")

        if fire_count > 0:
            st.error(f"🔥 YANGIN UYARISI! {fire_count} bölgede yangın tespit edildi!")
        if smoke_count > 0:
            st.warning(f"💨 DUMAN UYARISI! {smoke_count} bölgede duman tespit edildi!")

        st.markdown("### Detaylı Tespit Listesi")
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])

            icon = "🔥" if label == "fire" else "💨"
            color = "red" if label == "fire" else "orange"

            st.markdown(
                f"{icon} **#{i+1}** | "
                f"**Algılanan:** `{label.upper()}` | "
                f"**Güven Oranı:** `%{confidence * 100:.2f}`"
            )
    else:
        st.success("✅ Sistem Durumu: Temiz. Herhangi bir tehlike algılanmadı.")
        st.info(f"💡 Güven eşiği: %{confidence_threshold*100:.0f} — Tespit göremiyorsanız solda bulunan güven eşiğini düşürünüz.")