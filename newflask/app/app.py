import os
import sys
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Add the parent directory (newflask) to the Python path
# to allow imports from model_code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_code.detector import load_model, predict_image

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'blur_jpg_prob0.5.pth')
USE_CPU = True # Set to False to use GPU if available

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Model Loading ---
model = None
device = 'cpu'
model_loaded = False

print("Attempting to load model...")
if not os.path.exists(MODEL_WEIGHT_PATH):
    print("\n\n!!! WARNING: Model weight file not found !!!")
    print(f"Please download 'blur_jpg_prob0.5.pth' and place it in '{os.path.dirname(MODEL_WEIGHT_PATH)}'")
else:
    try:
        model, device = load_model(MODEL_WEIGHT_PATH, use_cpu=USE_CPU)
        model_loaded = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return "Model is not loaded due to missing weight file or an error during loading.", 500

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction
        probability = predict_image(model, device, filepath)

        # Determine result
        if probability is not None:
            if probability > 0.5:
                prediction_text = f"Prediction: Fake ({probability:.2%})"
            else:
                prediction_text = f"Prediction: Real ({1-probability:.2%})"
        else:
            prediction_text = "Could not process image."

        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template('index.html', model_loaded=model_loaded, prediction=prediction_text, image_file_url=image_url)

    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0')
