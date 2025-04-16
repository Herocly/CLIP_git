from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import os
import run

app = Flask(__name__)
CORS(app)

path_tmp = ""

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload():
    # 检查是否有文件上传
    text_file = request.form.get('text')


    if 'image' not in request.files:
        return jsonify({
            "success" : False,
            "error": "请上传图片文件"
            }), 400

    image_file = request.files['image']

    # 检查是否选择了文件
    if image_file.filename == '':
        return jsonify({
            "success": False,
            "error": "未选择文件"
        }), 400

    # 检查文件类型
    if not allowed_file(image_file.filename):
        return jsonify({
            "success": False,
            "error": "不支持的文件类型"
            }), 400

    try:
        # 验证图片文件
        image = Image.open(image_file.stream)
        image.verify()  # 验证图片完整性

        # 重置文件指针
        image_file.stream.seek(0)
        image = Image.open(image_file.stream)

        # 确保上传目录存在
        upload_dir = "./uploads/"
        os.makedirs(upload_dir, exist_ok=True)

        # 保存文件
        save_path = os.path.join(upload_dir, image_file.filename)
        image.save(save_path)
        path_tmp = save_path

        # return jsonify(run.read_with_post(path_tmp,
        #                               ["human",
        #                                "an animal",
        #                                "huge Buildings",
        #                                "cartoon",
        #                                "plants",
        #                                "disaster",
        #                                "AI",
        #                                "festival"]))
        return jsonify(run.zeroshot_strawberry_test(path_tmp))#将输入传到后端

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"图片处理失败: {str(e)}"
        }), 500


# @app.route('/result', methods=['GET'])
# def get_result():

#     try:
        

#     except Exception as e:
#         print(e)
#         return jsonify({
#             "success": False,
#             "error": f"错误码: {str(e)}"
#         }), 500



if __name__ == '__main__':
    # 创建上传目录
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")
    app.run(debug=True)