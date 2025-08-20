from flask import Flask, render_template, request, redirect, url_for
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 classes: Scab, Black Rot, Rust, Healthy
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
classes = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

# Preprocess function (matches training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess and predict
            img = Image.open(file_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                prediction = classes[predicted.item()]
            
            return render_template('index.html', prediction=prediction, image_path=url_for('static', filename='uploads/' + filename))
    
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)