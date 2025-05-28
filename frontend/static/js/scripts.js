// Handles both image and video preview
function previewFile(input) {
    const file = input.files[0];
    const imagePreview = document.getElementById("imagePreview");
    const videoPreview = document.getElementById("videoPreview");
    if (!file) return;

    // Hide both previews first
    if (imagePreview) imagePreview.style.display = "none";
    if (videoPreview) videoPreview.style.display = "none";

    if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = function (e) {
            if (imagePreview) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = "block";
            }
        };
        reader.readAsDataURL(file);
    } else if (file.type === "video/mp4") {
        if (videoPreview) {
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.style.display = "block";
        }
    }
}

async function sendImage() {
    const input = document.getElementById("imageInput");
    const result = document.getElementById("result");

    if (!input.files.length) {
        result.textContent = "Please select an image or video.";
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append("image", file);

    try {
        result.textContent = "Processing...";
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            if (data.video_summary) {
                // Video result
                const vs = data.video_summary;
                result.innerHTML =
                    `<b>Video Analysis:</b><br>
                    Majority: <b>${vs.majority_label}</b><br>
                    AI frames: ${(vs.ai_percent * 100).toFixed(1)}%<br>
                    Human frames: ${(vs.human_percent * 100).toFixed(1)}%<br>
                    Frames analyzed: ${vs.frame_count}`;
            } else if (data.prediction_label && data.probabilities) {
                // Image result
                result.innerHTML =
                    `<b>Prediction:</b> ${data.prediction_label}<br>
                    AI probability: ${(data.probabilities[1] * 100).toFixed(1)}%<br>
                    Human probability: ${(data.probabilities[0] * 100).toFixed(1)}%`;
            } else {
                result.textContent = "Prediction: " + (data.prediction || "Unknown");
            }
        } else {
            result.textContent = "Error: " + data.error;
        }
    } catch (error) {
        result.textContent = "Request failed.";
    }
}

// Optional: Attach previewFile to input change event
document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("imageInput");
    if (input) {
        input.addEventListener("change", function () {
            previewFile(input);
        });
    }
});
