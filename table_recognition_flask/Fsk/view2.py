from flask_bootstrap import Bootstrap
from flask import Flask, jsonify, send_from_directory, url_for
from flask import request
import cv2
import numpy as np
import os
import torch
import yaml
torch.set_num_threads(4)
from TableStructureRec import TableProcessor


from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)
bootstrap = Bootstrap(app)
app.config['JSON_AS_ASCII'] = False

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# 读取文件保存的路径
UPLOAD_FOLDER = '_upload_images/'
# 识别后文件输出的路径
OUTPUT_UPLOAD_FOLDER = 'Fsk/static/output_images'
OUTPUT_HTML = 'Fsk/static/output_htmls/output.html'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_UPLOAD_FOLDER'] = OUTPUT_UPLOAD_FOLDER
app.config['OUTPUT_HTML'] = OUTPUT_HTML

if not os.path.exists(OUTPUT_UPLOAD_FOLDER):
    os.makedirs(OUTPUT_UPLOAD_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processor = TableProcessor()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    result = {}
    try:
        if 'file' not in request.files:
            return '没有文件部分'
        file = request.files['file']
        if file.filename == '':
            return '没有选择文件'
        if file and file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if request.form.get('table_type') == 'lineless':
                image, html = processor.lineless_process_image(file_path)
                processor.write_html(app.config['OUTPUT_HTML'], html)
            elif request.form.get('table_type') == 'lined':
                image, html = processor.wired_process_image(file_path)
                processor.write_html(app.config['OUTPUT_HTML'], html)
            else:
                return '未指定表格类型'
            output_path = os.path.join(app.config['OUTPUT_UPLOAD_FOLDER'], filename)
            # read html
            with open(app.config['OUTPUT_HTML'], 'r', encoding='utf-8') as file:
                html_content = file.read()
            # write img
            cv2.imwrite(output_path, image)
            result['table_rec_Code'] = {'1': '成功返回结果'}

            return jsonify({'message': '识别成功', 'image_url': url_for('static', filename='output_images/' + filename), 'table_html': html_content })
        else:
            return '不支持的文件类型'
    except:
        return '网络计算过程未能正确返回结果'







