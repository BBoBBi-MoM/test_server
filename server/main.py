import os

import torch
from flask import Flask, render_template, request
from utils import draw_bbox, inference

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('./static/weights/batchsize16_size512_loss_20155.pt',
                   map_location=device)
app = Flask(__name__, static_folder='static')

@app.route('/')
def root():
    return render_template('input_page.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploades.'
    
    image = request.files['image']
    img_path = os.path.join('./static/uploads/', image.filename)
    save_path = os.path.join('./static/outputs/', image.filename)
    image.save(img_path)
    inference(img_path, save_path ,model, device, 0.8)
    return render_template('output_page.html', 
                           filename=str(image.filename),
                           image_url=save_path)

if __name__ == '__main__':
    app.run(debug=True)