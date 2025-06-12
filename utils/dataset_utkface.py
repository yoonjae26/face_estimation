import os
import shutil

# 🔁 Thư mục gốc chứa ảnh UTKFace
src_folder = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\data\output"
male_folder = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\data\male"
female_folder = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\data\female"

# 📁 Tạo thư mục đích nếu chưa tồn tại
os.makedirs(male_folder, exist_ok=True)
os.makedirs(female_folder, exist_ok=True)

def parse_gender(filename):
    try:
        # Ví dụ: 24_1_0_20170116174525125.jpg
        return int(filename.split("_")[1])
    except:
        return -1  # Trường hợp lỗi định dạng

count_male = 0
count_female = 0

# 🚀 Bắt đầu phân loại
for filename in os.listdir(src_folder):
    if not filename.endswith(".jpg"):
        continue

    gender = parse_gender(filename)
    src_path = os.path.join(src_folder, filename)

    if gender == 0:
        shutil.copy(src_path, os.path.join(male_folder, filename))
        count_male += 1
    elif gender == 1:
        shutil.copy(src_path, os.path.join(female_folder, filename))
        count_female += 1
    else:
        print(f"❌ Bỏ qua ảnh không xác định: {filename}")

print(f"✅ Đã chia xong: {count_male} ảnh nam, {count_female} ảnh nữ")