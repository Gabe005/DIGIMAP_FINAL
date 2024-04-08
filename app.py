from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2 as cv
from cv2 import dnn_superres
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def enhance_image(image_path):
    logger.debug('Enhancing image: %s', image_path)

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

    logger.debug('Image enhancement complete')

    return upscaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug('Received upload request')
    if 'imageFile' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'})

    file = request.files['imageFile']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file:
        img_path = os.path.join(app.instance_path, file.filename)
        logger.debug('Saving uploaded file to %s', img_path)
        file.save(img_path)

        enhanced_img = enhance_image(img_path)

        # Save enhanced image to a temporary file
        enhanced_img_path = os.path.join(app.instance_path, 'enhanced_' + file.filename)
        logger.debug('Saving enhanced image to %s', enhanced_img_path)
        cv.imwrite(enhanced_img_path, enhanced_img)

        logger.debug('Upload processing complete')
        return jsonify({'enhanced_image_path': enhanced_img_path})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.instance_path, filename), as_attachment=True)