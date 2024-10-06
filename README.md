# Pixel Art Sprite AI Generator

Dự án này nhằm xây dựng một AI có khả năng tạo và xử lý tất cả các sprite liên quan đến nhân vật trong game 2D. Mục tiêu là tạo ra một công cụ mạnh mẽ hỗ trợ các nhà phát triển và nghệ sĩ trong việc tạo, chỉnh sửa, và hoạt động với sprite trong quá trình phát triển game. AI được thiết kế dựa trên Generative Adversarial Networks (GANs) kết hợp với các kỹ thuật tiên tiến khác để tạo ra sprite chất lượng cao với nhiều hoạt động khác nhau.

## Chức Năng Chính

- **Tạo Sprite Nhân Vật Mới**: Tạo ra các sprite nhân vật mới dựa trên mô tả hoặc hình ảnh đầu vào.
- **Tạo Hướng Nhìn Khác Nhau**: Tự động tạo ra các hướng nhìn của nhân vật (trước, sau, trái, phải) dựa trên một sprite đơn lẻ.
- **Tạo Chuyển Động**: Sinh ra các chuyển động cho nhân vật, như đi, chạy, nhảy, cúi xuống, v.v., từ sprite cơ bản.
- **Tùy Biến Phong Cách Pixel Art**: Tự động thêm màu sắc, trang phục, và phụ kiện cho nhân vật theo nhiều phong cách khác nhau.
- **API Tích Hợp**: Sử dụng API để cung cấp khả năng tạo và chỉnh sửa sprite từ các ứng dụng khác.

## Công Cụ Và Thư Viện Chính

- **Python 3.7+**
- **PyTorch**: Thư viện học sâu để xây dựng và huấn luyện mô hình GAN.
- **NumPy**: Thư viện xử lý số liệu.
- **Pillow**: Thư viện xử lý hình ảnh.
- **OpenCV**: Tiền xử lý hình ảnh và phát hiện đặc trưng.
- **Flask**: Tạo API để tích hợp AI với các ứng dụng khác.

  ```bash
     python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 numpy pillow flask
  ```

2. Ghi các thư viện đã cài đặt vào requirements.txt
   ```bash
      python -m pip freeze > requirements.txt
   ```

## Cài Đặt

1. Tạo và kích hoạt môi trường ảo:

   ```bash
   python -m venv pixelart-env
   source pixelart-env/bin/activate  # Trên Windows: pixelart-env\Scripts\activate
   ```

2. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```

## Tiền Xử Lý Dữ Liệu

1. Chuẩn Bị Dữ Liệu
   Đặt các hình ảnh thô vào thư mục `data/raw/`.

2. Tiền Xử Lý Dữ Liệu
   Chạy script để chuyển đổi hình ảnh thành định dạng phù hợp với mô hình:
   ```bash
   python -m scripts.preprocess
   ```

## Huấn Luyện Mô Hình

- Chạy script huấn luyện mô hình GAN:

```bash
python scripts/train.py
```

## Triển Khai API

1. Chạy API
   Chạy Flask server để triển khai API tạo pixel art:
   ```bash
   python app/api.py
   ```

- API sẽ chạy tại `http://127.0.0.1:5000`.

2. Sử Dụng API
   Gửi yêu cầu POST đến endpoint /generate với payload JSON chứa mô tả văn bản. Ví dụ:
   ```bash
   {
   "description": "A vibrant 16x16 pixel art of a sunset"
   }
   ```

- Lưu ý: Mô tả văn bản được chuyển đổi thành vector đầu vào cho Generator (các phương pháp cụ thể có thể cần tinh chỉnh).

## Cấu Trúc Thư Mục

- **data/**: Chứa dữ liệu hình ảnh thô (raw/) và dữ liệu đã xử lý (processed/).
- **models/**: Chứa mã nguồn định nghĩa mô hình GAN (Generator, Discriminator, và GAN).
- **scripts/**: Chứa các script cho tiền xử lý dữ liệu và huấn luyện mô hình.
- **app/**: Chứa mã nguồn triển khai API với Flask.
- **requirements.txt**: Danh sách các thư viện Python cần thiết.
- **.gitignore**: Danh sách các tệp và thư mục không đưa vào phiên bản.
