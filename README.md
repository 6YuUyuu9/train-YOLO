# Dog Breed Segmentation (client-side)

YOLO26n-seg (polygon weight) รันในเบราว์เซอร์ด้วย onnxruntime-web ไม่มี server

## ไฟล์
- `index.html`, `app.js`: หน้าเว็บและโค้ด preprocess / NMS / mask
- `model.onnx`: export จาก `runs/segment/polygon/weights/best.pt`
- `ort.wasm.min.js`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm`: onnxruntime-web 1.31.1
- `examples/`: ภาพตัวอย่างจาก test set

## อัปเดตโมเดลหลังเทรนใหม่
```python
from ultralytics import YOLO
YOLO("runs/segment/polygon/weights/best.pt").export(format="onnx", imgsz=640)
# คัดลอก best.onnx มาทับ webapp_js/model.onnx
```

## ทดลองในเครื่อง
ต้องเปิดผ่าน web server (ดับเบิลคลิก index.html ตรง ๆ จะโหลดโมเดลไม่ได้)
```
cd webapp_js
python -m http.server 8000
```
แล้วเปิด http://localhost:8000

## ทำลิงก์สาธารณะด้วย GitHub Pages
1. สร้าง repo ใหม่ อัปโหลดทุกไฟล์ในโฟลเดอร์นี้
2. Settings > Pages > Source: Deploy from a branch, Branch: main / root
3. รอ 1-2 นาที จะได้ลิงก์ `https://<username>.github.io/<repo>/`
