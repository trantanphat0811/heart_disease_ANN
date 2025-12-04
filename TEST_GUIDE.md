# 🧪 Hướng Dẫn Test Liên Kết Dữ Liệu

## Tình Trạng Hiện Tại

✅ **Hệ thống đã hoàn thành:**
- Dữ liệu từ CSV tự động được lưu vào `localStorage`
- Dữ liệu lưu thành 2 collection:
  - `patient_history`: Lịch sử batch
  - `all_patients`: Danh sách tất cả bệnh nhân

## 3 Cách Test

### **Cách 1: Trang Test Tự Động (⭐ Khuyến Nghị)**

1. Mở: http://localhost:3000/static/test-full-flow.html
2. Nhấn **"🚀 Giả Lập Upload"** - Tạo dữ liệu test
3. Nhấn **"📊 Kiểm Tra"** - Xem dữ liệu được lưu
4. Mở 2 tab mới:
   - http://localhost:3000/static/patients.html (Xem bệnh nhân)
   - http://localhost:3000/static/history.html (Xem lịch sử)

### **Cách 2: Upload CSV Thực Tế**

1. Mở: http://localhost:3000/static/batch_check.html
2. Upload file CSV (hoặc tải mẫu)
3. Xem kết quả dự đoán
4. ✅ Dữ liệu **tự động lưu** vào localStorage
5. Mở: http://localhost:3000/static/patients.html
   - ✅ Sẽ thấy bệnh nhân vừa upload
6. Mở: http://localhost:3000/static/history.html
   - ✅ Sẽ thấy batch vừa tạo

### **Cách 3: Debug Chi Tiết**

1. Mở DevTools: F12 → Console tab
2. Kiểm tra từng bước:
   ```javascript
   // Xem statistics
   PatientDataManager.getStatistics()
   
   // Xem tất cả bệnh nhân
   PatientDataManager.getAllPatients()
   
   // Xem lịch sử batch
   PatientDataManager.getBatchHistory()
   ```

## 📍 Các Trang Để Test

| Trang | URL | Mô Tả |
|------|-----|-------|
| 🧪 **Test Full Flow** | `/static/test-full-flow.html` | **Bắt đầu từ đây** - Giả lập & kiểm tra |
| 📊 **Dự Đoán Hàng Loạt** | `/static/batch_check.html` | Upload CSV thực tế |
| 👥 **Quản Lý Bệnh Nhân** | `/static/patients.html` | Xem danh sách bệnh nhân |
| 📜 **Lịch Sử** | `/static/history.html` | Xem lịch sử batch |
| 🔍 **Debug Storage** | `/static/test-storage.html` | Chi tiết localStorage |

## 🎯 Kỳ Vọng Kết Quả

### **Sau Upload CSV:**

✅ **Trên batch_check.html:**
- Hiển thị kết quả dự đoán
- Nút "✅ Đã lưu X bệnh nhân vào hệ thống"
- Console: `✅ Batch saved with ID: batch_...`

✅ **Trên patients.html:**
- Auto-load danh sách bệnh nhân
- Hiển thị tất cả bệnh nhân từ batch
- Cho phép tìm kiếm, xóa, xuất

✅ **Trên history.html:**
- Auto-load lịch sử batch
- Hiển thị thông tin batch (file, date, stats)
- Cho phép xem chi tiết, xóa, xuất

## 🐛 Troubleshooting

### **Vấn Đề: Dữ liệu không hiển thị trên patients.html**

1. **Kiểm tra Console (F12):**
   ```
   ✅ [patients.html] loadPatients called
   ✅ [patients.html] Loaded X patients
   ```
   - Nếu thấy, dữ liệu được load
   - Nếu không, kiểm tra step 2 bên dưới

2. **Kiểm tra localStorage:**
   - DevTools → Application → Local Storage
   - Tìm: `all_patients` key
   - Nếu không có, dữ liệu chưa được lưu

3. **Kiểm tra API call:**
   - DevTools → Network tab
   - Upload CSV
   - Tìm request `POST /predict/batch`
   - Response phải có `results` array

### **Vấn Đề: PatientDataManager not defined**

1. Reload trang (Ctrl+R)
2. Kiểm tra script import: `<script src="/static/js/patient-data.js">`
3. Verify file exists: http://localhost:3000/static/js/patient-data.js

### **Vấn Đề: Server không chạy**

```bash
# Kiểm tra
lsof -i :3000

# Restart
pkill -f python3
cd /Users/trantanphat/Documents/Python/heart_disease
python3 run_server.py &
```

## ✨ Chức Năng Hoàn Chỉnh

### **batch_check.html:**
- 📤 Upload CSV
- 📊 Dự đoán hàng loạt
- 💾 **Tự động lưu** vào localStorage
- 🔄 Tải lại
- 🗑️ Xóa dữ liệu

### **patients.html:**
- 👥 Danh sách bệnh nhân (load từ localStorage)
- 🔍 Tìm kiếm
- 👁️ Xem chi tiết
- 🗑️ Xóa bệnh nhân
- 📥 Xuất CSV
- 📊 Thống kê

### **history.html:**
- 📜 Lịch sử batch (load từ localStorage)
- 👁️ Xem chi tiết batch
- 🗑️ Xóa batch
- 📥 Xuất CSV/JSON
- 📊 Thống kê tổng hợp

## 🚀 Bắt Đầu

**Nhấn vào link này để bắt đầu test:**

👉 **http://localhost:3000/static/test-full-flow.html**

