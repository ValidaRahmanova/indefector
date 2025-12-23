import cv2
from ultralytics import YOLO

model_detect = YOLO("bottle_defect_model.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model_detect(frame, imgsz=640, conf=0.01, iou=0.7)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label_orig = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 1. Qapaq yoxlaması
        if label_orig == "cap":
            if conf < 0.50: continue # Qapaqdan 50% əmin deyilsə göstərmə
            label = f"Qapaq hissesi {conf:.2f}"
            color = (255, 0, 0) # Yaşıl
            thickness = 2

        # 2. Qapaqsızlıq yoxlaması
        elif label_orig == "no-cap":
            label = "Qapaqsiz!"
            color = (0, 0, 255) # Qırmızı
            thickness = 3

        # 3. Defekt (Əzik) yoxlaması - ƏN HƏSSAS HİSSƏ
        elif label_orig == "crumbled":
            # Hətta 10% (0.10) ehtimal olsa belə defektli de
            label = f"DEFEKTLI {conf:.2f}"
            color = (0, 0, 255) # Qırmızı
            thickness = 3

        # 4. Normal şüşə yoxlaması
        elif label_orig == "not-crumbled":
            # Əgər model 85% əmin deyilsə, "normaldır" deməsin (defekti qaçırmasın)
            if conf < 0.95: continue 
            label = f"Normal {conf:.2f}"
            color = (0, 255, 0) # Yaşıl
            thickness = 1
        
        else:
            label = label_orig
            color = (255, 255, 255)
            thickness = 1

        # Ekrana çəkmə
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Bottle Quality Control", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()