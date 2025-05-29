from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import tempfile
import os
import imageio

app = Flask(__name__)
# Restrict CORS to your GitHub Pages domain for security
CORS(app, origins=["https://druiidr.github.io/TIMGA/"])

# Load your Keras model
model = load_model("models/detector model v2.1.2(curr).h5")

# Define your class labels (ensure this matches your model's training order)
class_labels = ["Human", "AI"]

# Image transform: always resize to 256x256
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if your model expects it
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image or video provided"}), 400

        file = request.files["image"]
        filename = file.filename.lower()

        # Handle video (mp4) with imageio
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
                return jsonify({"success": False, "error": "Could not read video frames."}), 400

            num_samples = min(30, frame_count)
            sample_indices = set(np.linspace(0, frame_count - 1, num_samples, dtype=int))
            ai_count = 0
            human_count = 0
            ai_probs = []
            human_probs = []
            total = 0

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

            ai_percent = ai_count / total if total else 0
            human_percent = human_count / total if total else 0
            majority_label = "AI" if ai_count > human_count else "Human"

            return jsonify({
                "success": True,
                "video_summary": {
                    "ai_percent": ai_percent,
                    "human_percent": human_percent,
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
            probabilities = preds[0].tolist()
            return jsonify({
                "success": True,
                "prediction_index": prediction_index,
                "prediction_label": prediction_label,
                "probabilities": probabilities
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
