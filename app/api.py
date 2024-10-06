from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import io
from models.generator import Generator

# Khởi tạo Flask
app = Flask(__name__)

# Load mô hình Generator
generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()  # Đặt mô hình sang chế độ đánh giá (evaluation)

# Định nghĩa route cho API
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Lấy đầu vào (ví dụ từ JSON hoặc file gửi đến)
        input_data = request.json.get('input_data', None)

        # Kiểm tra dữ liệu đầu vào
        if input_data is None:
            return jsonify({"error": "No input data provided"}), 400

        # Tạo đầu vào cho Generator từ dữ liệu đầu vào
        # Ví dụ nếu input là nhiễu ngẫu nhiên:
        z = torch.randn(1, 4, 64, 64)  # Điều chỉnh theo yêu cầu của mô hình
        with torch.no_grad():
            generated_image = generator(z)  # Sinh ảnh mới từ Generator
        
        # Convert tensor thành image để hiển thị
        image = ToPILImage()(generated_image.squeeze(0))

        # Lưu ảnh vào buffer
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        # Gửi file ảnh về cho client
        return send_file(buf, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)