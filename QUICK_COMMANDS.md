# 🚀 QUICK START COMMANDS

## Hệ Thống Dự Đoán Bệnh Tim Bằng CNN (Heart Disease Prediction with Deep Learning)

---

## 📋 LỆNH BẮT ĐẦU NHANH

### **1️⃣ Bước 1: Thiết Lập Môi Trường (Setup)**

```bash
# Kiểm tra phiên bản Python (yêu cầu 3.8+)
python3 --version

# Cài đặt dependencies
pip install -r requirements.txt

# Xác nhận cài đặt thành công
pip list | grep -E "torch|fastapi|pandas|pillow"
```

---

### **2️⃣ Bước 2: Tạo Thư Mục & Dữ Liệu (Initialize)**

```bash
# Chạy script thiết lập nhanh
python3 quick_setup.py

# Kết quả:
# ✓ Tạo các thư mục cần thiết
# ✓ Kiểm tra dependencies
# ✓ Tạo 10 hình ảnh ECG mẫu
# ✓ Hiển thị hướng dẫn tiếp theo
```

---

### **3️⃣ Bước 3: Huấn Luyện Mô Hình CNN (Training)**

```bash
# Bắt đầu huấn luyện ResNet50
python3 train_cnn.py

# Thông tin huấn luyện:
# ⏱️  Thời gian: ~15-30 phút (CPU) hoặc ~2-5 phút (GPU)
# 🎯 Mô hình: ResNet50 (Transfer Learning)
# 📊 Epochs: 50, Batch Size: 32
# 💾 Lưu tại: models/cnn_model.pth

# Với GPU (nếu có sẵn):
CUDA_VISIBLE_DEVICES=0 python3 train_cnn.py
```

---

### **4️⃣ Bước 4: Khởi Động Server (Start Server)**

```bash
# Khởi động FastAPI server trên port 3000
python3 run_server.py

# Hoặc tùy chỉnh host/port:
HOST=127.0.0.1 PORT=3000 python3 run_server.py

# Server sẽ chạy trên:
# 🌐 http://127.0.0.1:3000
# 🌐 http://localhost:3000
```

---

### **5️⃣ Bước 5: Mở Trình Duyệt Web (Access Web UI)**

```bash
# Trên Mac
open http://localhost:3000

# Trên Linux
xdg-open http://localhost:3000

# Trên Windows
start http://localhost:3000

# Hoặc nhập URL vào trình duyệt:
http://localhost:3000/menu.html
```

---

## 🌐 WEB PAGES & FEATURES

| Trang | URL | Chức Năng |
|-------|-----|----------|
| 📊 Trang Chủ | `http://localhost:3000/menu.html` | Giao diện chính, menu chức năng |
| 🔮 Dự Đoán Đơn | `http://localhost:3000/train.html` | Nhập thông tin bệnh nhân, nhận dự đoán |
| 📷 Dự Đoán Ảnh | `http://localhost:3000/image_predict.html` | Tải lên hình ảnh ECG, dự đoán |
| 📊 Batch Check | `http://localhost:3000/batch_check.html` | Tải CSV, dự đoán hàng loạt |
| 👥 Bệnh Nhân | `http://localhost:3000/patients.html` | Quản lý thông tin bệnh nhân |
| 📈 Báo Cáo | `http://localhost:3000/reports.html` | Xem báo cáo & thống kê |
| 📜 Lịch Sử | `http://localhost:3000/history.html` | Xem lịch sử dự đoán |

---

## 🔌 API ENDPOINTS

```bash
# 1. Kiểm tra server
curl http://127.0.0.1:3000/health

# 2. Dự đoán từ dữ liệu bệnh nhân (POST)
curl -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": 1, "weight": 70, ...}'

# 3. Dự đoán hàng loạt từ CSV (POST)
curl -X POST http://127.0.0.1:3000/predict/batch \
  -F "file=@data.csv"

# 4. Dự đoán từ hình ảnh ECG (POST)
curl -X POST http://127.0.0.1:3000/predict/image \
  -F "file=@ecg_image.png"
```

---

## 🎯 ADVANCED COMMANDS

### Kiểm Tra & Debugging

```bash
# Xem trạng thái server
curl -v http://127.0.0.1:3000/health

# Xem phiên bản PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Kiểm tra GPU (CUDA)
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Xem thông tin package
pip show torch

# Liệt kê các dependencies
pip freeze | grep -E "torch|fastapi|pandas"
```

### Quản Lý Mô Hình

```bash
# Xóa mô hình cũ (để train lại từ đầu)
rm -f models/cnn_model.pth models/cnn_model_history.json

# Xem thông tin mô hình
python3 -c "import torch; model = torch.load('models/cnn_model.pth'); print(model)"

# Backup mô hình
cp models/cnn_model.pth models/cnn_model_backup.pth

# So sánh hai mô hình
ls -lh models/cnn_model*.pth
```

### Quản Lý Dữ Liệu

```bash
# Thêm hình ảnh ECG mới
cp /path/to/ecg/images/* data/train/healthy/
# hoặc
cp /path/to/ecg/images/* data/train/disease/

# Kiểm tra số lượng hình ảnh
ls data/train/healthy/ | wc -l
ls data/train/disease/ | wc -l

# Xem dữ liệu CSV
head -5 data/synthetic_heart_disease_dataset.csv

# Đếm số dòng CSV
wc -l data/synthetic_heart_disease_dataset.csv
```

### Server & Processes

```bash
# Dừng server (Ctrl+C trong terminal)
# Hoặc từ terminal khác:
kill %1  # hoặc lấy PID và kill

# Kiểm tra port đang sử dụng
lsof -i :3000

# Khởi động lại server
pkill -f run_server.py
python3 run_server.py

# Background process
nohup python3 run_server.py > server.log 2>&1 &

# Xem log real-time
tail -f server.log
```

---

## 📁 THƯ MỤC QUAN TRỌNG

```
heart_disease/
├── data/                                    # Dữ liệu
│   ├── train/                              # Ảnh huấn luyện
│   │   ├── healthy/                        # ECG bình thường
│   │   └── disease/                        # ECG bất thường
│   ├── val/                                # Ảnh kiểm định
│   └── synthetic_heart_disease_dataset.csv # CSV dữ liệu
│
├── models/                                  # Mô hình
│   ├── cnn_model.pth                       # Mô hình CNN (chính)
│   ├── cnn_model_history.json              # Thống kê huấn luyện
│   └── best_model.joblib                   # Mô hình scikit-learn (cũ)
│
├── web/                                     # Giao diện web
│   ├── menu.html                           # Trang chủ
│   ├── train.html                          # Dự đoán đơn
│   ├── batch_check.html                    # Batch prediction
│   ├── image_predict.html                  # Dự đoán từ ảnh
│   ├── patients.html                       # Quản lý bệnh nhân
│   ├── reports.html                        # Báo cáo
│   └── history.html                        # Lịch sử
│
├── src/                                     # Backend code
│   └── app.py                              # FastAPI server
│
├── uploads/                                 # Upload files
│   ├── images/                             # Tải lên ảnh
│   └── results/                            # Kết quả export
│
├── logs/                                    # Logs
│   └── training.log                        # Log huấn luyện
│
├── train_cnn.py                            # Script huấn luyện CNN
├── quick_setup.py                          # Script thiết lập
├── run_server.py                           # Script khởi động server
├── requirements.txt                        # Dependencies
├── README_RUN.md                           # Hướng dẫn chạy
└── QUICK_COMMANDS.md                       # File này
```

---

## ⚠️ TROUBLESHOOTING

### Vấn Đề: Lỗi Import PyTorch

```bash
# Giải pháp:
pip install --upgrade torch torchvision

# Kiểm tra:
python3 -c "import torch; print(torch.__version__)"
```

### Vấn Đề: Port 3000 đang sử dụng

```bash
# Xem process sử dụng port
lsof -i :3000

# Dừng process
kill -9 <PID>

# Hoặc dùng port khác:
PORT=8081 python3 run_server.py
```

### Vấn Đề: Không tìm thấy tệp CSV

```bash
# Kiểm tra vị trí file:
ls -la data/synthetic_heart_disease_dataset.csv

# Nếu không có, tải từ nguồn hoặc tạo sample
# Đảm bảo file có 21 cột với delimiter là TAB
```

### Vấn Đề: Lỗi Permission Denied

```bash
# Cấp quyền execute:
chmod +x train_cnn.py
chmod +x quick_setup.py
chmod +x run_server.py
```

---

## 📚 TÀI LIỆU THÊM

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **ResNet Paper**: https://arxiv.org/abs/1512.03385

---

## 🎯 WORKFLOW KHUYẾN NGHỊ

```
1. python3 quick_setup.py
   ↓
2. pip install -r requirements.txt
   ↓
3. python3 train_cnn.py (Chờ ~15-30 phút)
   ↓
4. python3 run_server.py (Mở terminal khác)
   ↓
5. open http://localhost:3000
   ↓
6. Sử dụng hệ thống
   ↓
7. Nhấn Ctrl+C để dừng server
```

---

## ❓ CÂU HỎI THƯỜNG GẶP

**Q: Làm sao để tăng tốc độ huấn luyện?**  
A: Dùng GPU (CUDA): `CUDA_VISIBLE_DEVICES=0 python3 train_cnn.py`

**Q: Làm sao để sử dụng dữ liệu hình ảnh của riêng tôi?**  
A: Copy vào `data/train/healthy/` và `data/train/disease/`, sau đó chạy `python3 train_cnn.py`

**Q: Mô hình được lưu ở đâu?**  
A: `models/cnn_model.pth` (PyTorch) và `models/best_model.joblib` (scikit-learn)

**Q: Làm sao để xem lịch sử dự đoán?**  
A: Mở `http://localhost:3000/history.html`

---

**Được tạo bởi: AI Copilot**  
**Ngày: December 2, 2024**  
**Phiên bản: 1.0**
