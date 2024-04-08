from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2 as cv
from cv2 import dnn_superres

app = Flask(__name__)
CORS(app)

def enhance_image(image_path):
    # initialize super resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # read the model
    path = 'EDSR_x4.pb'
    sr.readModel(path)

    # set the model and scale
    sr.setModel('edsr', 4)

    # Read the image
    image = cv.imread(image_path)

    # upsample the image
    upscaled = sr.upsample(image)

    """image = cv.imread(image_path)
    bicubic = cv.resize(image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv.INTER_CUBIC)"""

    return upscaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img_path = os.path.join(app.instance_path, file.filename)
        file.save(img_path)

        enhanced_img = enhance_image(img_path)

        # Save enhanced image to a temporary file
        enhanced_img_path = os.path.join(app.instance_path, 'enhanced_' + file.filename)
        cv.imwrite(enhanced_img_path, enhanced_img)

        return jsonify({'enhanced_image_path': enhanced_img_path})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.instance_path, filename), as_attachment=True)