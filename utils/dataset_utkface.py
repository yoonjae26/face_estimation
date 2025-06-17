import os
import shutil

# Constants
SRC_FOLDER = os.getenv("SRC_FOLDER", "data/source")
MALE_FOLDER = os.getenv("MALE_FOLDER", "data/male")
FEMALE_FOLDER = os.getenv("FEMALE_FOLDER", "data/female")

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(MALE_FOLDER, exist_ok=True)
os.makedirs(FEMALE_FOLDER, exist_ok=True)

def parse_gender(filename):
    """
    Phân tích giới tính từ tên file.
    Tham số:
        filename (str): Tên file theo định dạng 'tuổi_giới_tính_dân_tộc_ngày.jpg'
    Trả về:
        int: 0 cho nam, 1 cho nữ, -1 cho định dạng không hợp lệ
    """
    try:
        return int(filename.split("_")[1])
    except ValueError:
        return -1

def classify_images(src_folder=SRC_FOLDER, male_folder=MALE_FOLDER, female_folder=FEMALE_FOLDER):
    """
    Phân loại ảnh vào thư mục nam và nữ.
    Tham số:
        src_folder (str): Thư mục nguồn chứa ảnh
        male_folder (str): Thư mục đích cho ảnh nam
        female_folder (str): Thư mục đích cho ảnh nữ
    """
    count_male = 0
    count_female = 0

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

    print(f"✅ Đã phân loại xong: {count_male} ảnh nam, {count_female} ảnh nữ")

if __name__ == "__main__":
    classify_images()