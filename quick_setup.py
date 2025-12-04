#!/usr/bin/env python3
"""
Quick Setup Script for Heart Disease CNN Prediction System
Thiết lập nhanh hệ thống dự đoán bệnh tim bằng CNN
"""

import os
import sys
import shutil
from pathlib import Path

def print_header(title):
    """In tiêu đề với style"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def create_directories():
    """Tạo các thư mục cần thiết"""
    print("1️⃣  Tạo các thư mục...")
    directories = [
        "data/",
        "data/train/",
        "data/train/healthy/",
        "data/train/disease/",
        "data/val/",
        "data/val/healthy/",
        "data/val/disease/",
        "data/test/",
        "models/",
        "models/cnn/",
        "logs/",
        "uploads/",
        "uploads/images/",
        "uploads/results/",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Đã tạo thư mục: {directory}")
    print()

def check_dependencies():
    """Kiểm tra các package cần thiết"""
    print("2️⃣  Kiểm tra dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pillow': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'opencv': 'OpenCV',
        'scikit-learn': 'Scikit-learn',
    }
    
    missing_packages = []
    for pkg, name in required_packages.items():
        try:
            __import__(pkg)
            print(f"   ✓ {name} đã được cài đặt")
        except ImportError:
            print(f"   ✗ {name} chưa được cài đặt")
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"\n⚠️  Cảnh báo: Các package chưa được cài đặt: {', '.join(missing_packages)}")
        print("   Chạy: pip install -r requirements.txt\n")
    else:
        print("   ✓ Tất cả dependencies đã sẵn sàng\n")
    
    return len(missing_packages) == 0

def create_sample_images():
    """Tạo hình ảnh mẫu để kiểm tra"""
    print("3️⃣  Tạo hình ảnh mẫu (ECG tổng hợp)...")
    
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Tạo 10 hình ảnh healthy ECG
        for i in range(5):
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            # Vẽ sóng ECG bình thường (xanh lá)
            points = []
            for x in range(224):
                y = 112 + 40 * np.sin(x / 20) + 20 * np.cos(x / 40)
                points.append((x, int(y)))
            
            for j in range(len(points) - 1):
                draw.line([points[j], points[j+1]], fill=(34, 139, 34), width=2)
            
            img.save(f"data/train/healthy/ecg_healthy_{i}.png")
        
        # Tạo 10 hình ảnh disease ECG
        for i in range(5):
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            # Vẽ sóng ECG bất thường (đỏ)
            points = []
            for x in range(224):
                y = 112 + 60 * np.sin(x / 15) + 30 * np.cos(x / 25) + np.random.randint(-10, 10)
                points.append((x, int(y)))
            
            for j in range(len(points) - 1):
                draw.line([points[j], points[j+1]], fill=(220, 20, 60), width=2)
            
            img.save(f"data/train/disease/ecg_disease_{i}.png")
        
        print("   ✓ Đã tạo 10 hình ảnh ECG mẫu (5 healthy, 5 disease)")
        print("   📂 Vị trí: data/train/healthy/ và data/train/disease/\n")
        return True
        
    except Exception as e:
        print(f"   ✗ Lỗi tạo hình ảnh mẫu: {e}")
        print("   ℹ️  Bạn có thể tải hình ảnh ECG thực tế vào thư mục data/ sau đó chạy training\n")
        return False

def print_next_steps():
    """In các bước tiếp theo"""
    print_header("🎉 ĐÃ HOÀN THÀNH THIẾT LẬP NHANH")
    
    print("📋 CÁC BƯỚC TIẾP THEO:\n")
    
    print("┌─ 1️⃣  HUẤN LUYỆN MÔ HÌNH CNN")
    print("│  Lệnh:")
    print("│  $ python3 train_cnn.py")
    print("│")
    print("│  ⏱️  Thời gian: ~15-30 phút (CPU) hoặc ~2-5 phút (GPU)")
    print("│  📊 Mô hình sẽ lưu tại: models/cnn_model.pth")
    print("│  💾 Thống kê huấn luyện: models/cnn_model_history.json")
    print("│")
    print("│  Tính năng:")
    print("│  ✓ Transfer Learning với ResNet50")
    print("│  ✓ 50 epochs huấn luyện")
    print("│  ✓ Hỗ trợ GPU (CUDA) nếu có sẵn")
    print("│  ✓ Lưu mô hình tốt nhất tự động")
    print("│  ✓ Hiển thị metrics chi tiết")
    print("└────────────────────────────────────────\n")
    
    print("┌─ 2️⃣  KHỞI ĐỘNG HỆ THỐNG")
    print("│  Lệnh:")
    print("│  $ python3 run_server.py")
    print("│")
    print("│  🌐 Server FastAPI sẽ chạy trên:")
    print("│  → http://127.0.0.1:8080")
    print("│  → http://localhost:8080")
    print("│")
    print("│  API Endpoints:")
    print("│  • POST /predict        - Dự đoán từ dữ liệu bệnh nhân")
    print("│  • POST /predict/batch  - Dự đoán hàng loạt từ CSV")
    print("│  • POST /predict/image  - Dự đoán từ hình ảnh ECG")
    print("│  • GET  /health        - Kiểm tra trạng thái server")
    print("└────────────────────────────────────────\n")
    
    print("┌─ 3️⃣  MỞ TRÌNH DUYỆT WEB")
    print("│  🌐 Truy cập các trang sau:\n")
    print("│  📊 Trang chủ:")
    print("│     http://localhost:8080/menu.html\n")
    print("│  🔮 Dự đoán đơn bệnh nhân:")
    print("│     http://localhost:8080/train.html")
    print("│     ➜ Nhập thông tin bệnh nhân, nhận kết quả dự đoán\n")
    print("│  📷 Dự đoán từ hình ảnh ECG:")
    print("│     http://localhost:8080/image_predict.html")
    print("│     ➜ Tải lên hình ảnh ECG, nhận kết quả dự đoán\n")
    print("│  📊 Kiểm tra hàng loạt từ CSV:")
    print("│     http://localhost:8080/batch_check.html")
    print("│     ➜ Tải lên file CSV, dự đoán nhiều bệnh nhân\n")
    print("│  👥 Quản lý bệnh nhân:")
    print("│     http://localhost:8080/patients.html\n")
    print("└────────────────────────────────────────\n")
    
    print("┌─ 4️⃣  THÊM DỮ LIỆU ẢNH THỰC TẾ (TÙY CHỌN)")
    print("│  Thư mục để lưu hình ảnh:")
    print("│  • data/train/healthy/  ← Ảnh ECG bình thường")
    print("│  • data/train/disease/  ← Ảnh ECG bất thường")
    print("│")
    print("│  Sau khi thêm ảnh:")
    print("│  $ python3 train_cnn.py  (chạy lại)")
    print("│")
    print("│  Định dạng hỗ trợ: .jpg, .png, .jpeg")
    print("│  Kích thước được phép: 224x224 pixel (tự động resize)")
    print("└────────────────────────────────────────\n")
    
    print("┌─ 5️⃣  DỪNG HỆ THỐNG")
    print("│  Nhấn: Ctrl+C")
    print("│")
    print("│  Server sẽ dừng một cách an toàn")
    print("└────────────────────────────────────────\n")
    
    print("┌─ 6️⃣  KIỂM TRA CÁC FILE QUAN TRỌNG")
    print("│  models/cnn_model.pth              ← Mô hình CNN đã huấn luyện")
    print("│  models/best_model.joblib          ← Mô hình CSV gốc")
    print("│  data/synthetic_heart_disease_dataset.csv  ← Dữ liệu CSV")
    print("│  requirements.txt                   ← Dependencies")
    print("│  train_cnn.py                       ← Script huấn luyện")
    print("│  run_server.py                      ← Script khởi động server")
    print("└────────────────────────────────────────\n")
    
    print_header("🚀 HỆ THỐNG SẴN SÀNG!")
    
    print("✨ LỆNH BẮT ĐẦU NGAY:")
    print("   1. python3 train_cnn.py")
    print("   2. python3 run_server.py")
    print("   3. open http://localhost:8080\n")
    
    print("📖 HƯỚNG DẪN CÓ THÊM TRONG:")
    print("   README_RUN.md")
    print("   train_cnn.py (dòng comment)")
    print("   src/app.py (API documentation)\n")

def main():
    """Hàm chính"""
    print_header("🏥 THIẾT LẬP HỆ THỐNG DỰ ĐOÁN BỆNH TIM BẰNG CNN")
    
    # 1. Tạo thư mục
    create_directories()
    
    # 2. Kiểm tra dependencies
    deps_ok = check_dependencies()
    
    # 3. Tạo hình ảnh mẫu
    images_created = create_sample_images()
    
    # 4. In các bước tiếp theo
    print_next_steps()
    
    print("="*60)
    print("💡 LỆNH CHÚ THÍCH NHANH:")
    print("="*60)
    print("""
# 🔧 Kiểm tra requirements
pip list | grep -E "torch|fastapi|pandas"

# 📦 Cài đặt lại dependencies (nếu cần)
pip install --upgrade -r requirements.txt

# 🧠 Huấn luyện mô hình CNN (Bước 1)
python3 train_cnn.py

# 🚀 Khởi động server (Bước 2)
python3 run_server.py

# 🌐 Mở web trên Mac
open http://localhost:8080

# 🌐 Mở web trên Linux
xdg-open http://localhost:8080

# 🌐 Mở web trên Windows
start http://localhost:8080

# 📊 Kiểm tra server đang chạy
curl http://127.0.0.1:8080/health

# 🔍 Xem chi tiết error log
tail -f logs/training.log

# 🧹 Xóa mô hình cũ (để train lại)
rm -f models/cnn_model.pth models/cnn_model_history.json

# 🐍 Chạy train_cnn.py với GPU (nếu có)
CUDA_VISIBLE_DEVICES=0 python3 train_cnn.py

# 📝 Xem các version của packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    """)
    print("="*60)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
