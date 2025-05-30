from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import tempfile
import os
import imageio

# Set static_folder to point to the frontend's static directory
app = Flask(__name__, static_folder='../frontend/static', static_url_path='')

# Safe path to model (important for both local and production environments)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "detector model v2.1.3(curr).h5")

# Load your Keras model
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ["Human", "AI"]

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image or video provided"}), 400

        file = request.files["image"]
        filename = file.filename.lower()

        # Handle video
        if filename.endswith('.mp4'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                file.save(temp_video.name)
                temp_video_path = temp_video.name

            reader = imageio.get_reader(temp_video_path, 'ffmpeg')
            try:
                frame_count = reader.count_frames()
            except Exception:
                reader.close()
                os.remove(temp_video_path)
                return jsonify({"success": False, "error": "Could not read video frames."}), 400

            if frame_count == 0:
                reader.close()
                os.remove(temp_video_path)
                return jsonify({"success": False, "error": "Video has no frames."}), 400

            num_samples = min(30, frame_count)
            sample_indices = set(np.linspace(0, frame_count - 1, num_samples, dtype=int))
            ai_count = human_count = total = 0
            ai_probs, human_probs = [], []

            for idx, frame in enumerate(reader):
                if idx in sample_indices:
                    pil_img = Image.fromarray(frame)
                    img_array = preprocess_image(pil_img)
                    preds = model.predict(img_array)
                    ai_prob = float(preds[0][class_labels.index("AI")])
                    human_prob = float(preds[0][class_labels.index("Human")])
                    ai_probs.append(ai_prob)
                    human_probs.append(human_prob)
                    if ai_prob > human_prob:
                        ai_count += 1
                    else:
                        human_count += 1
                    total += 1
                    if total >= num_samples:
                        break
            reader.close()
            os.remove(temp_video_path)

            majority_label = "AI" if ai_count > human_count else "Human"
            return jsonify({
                "success": True,
                "video_summary": {
                    "ai_percent": ai_count / total if total else 0,
                    "human_percent": human_count / total if total else 0,
                    "majority_label": majority_label,
                    "frame_count": total
                }
            })

        # Handle image
        else:
            img = Image.open(file).convert("RGB")
            img_array = preprocess_image(img)
            preds = model.predict(img_array)
            prediction_index = int(np.argmax(preds, axis=1)[0])
            prediction_label = class_labels[prediction_index] if prediction_index < len(class_labels) else f"Unknown ({prediction_index})"
            return jsonify({
                "success": True,
                "prediction_index": prediction_index,
                "prediction_label": prediction_label,
                "probabilities": preds[0].tolist()
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)