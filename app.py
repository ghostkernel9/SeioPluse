from flask import Flask, render_template, request, send_file, send_from_directory
from ultralytics import YOLO
import cv2
import os
import base64

app = Flask(__name__)

# Load model
model = YOLO(r"E:\Caner detecch\New folder\runs\detect\train9\weights\best.pt")

# ------------------ HOME ROUTE ------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------ SERVE CSS FROM TEMPLATES ------------------
@app.route("/style.css")
def serve_css():
    return send_from_directory("templates", "style.css")

# ------------------ PREDICT ROUTE ------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Save uploaded image
    input_path = "input.jpg"
    file.save(input_path)

    # Load image
    img = cv2.imread(input_path)

    # Run YOLO
    results = model(input_path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save and convert output
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, img)

    with open(output_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    return render_template(
        "index.html",
        message="Prediction Complete!",
        output_image=f"data:image/jpeg;base64,{encoded}"
    )

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)
