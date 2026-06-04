from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Başlangıç modelini (Nano) hafızaya yüklüyoruz
    model = YOLO("yolov8n.pt")
    
    # 2. Eğitimi Windows korumalı ve RTX 4050 ile başlatıyoruz
    model.train(
        data=r"C:\Users\ASUS\OneDrive\Desktop\FireDetectionProjesi\data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        workers=0  # Windows'un çökmesini engelleyen mucize parametre!
    )