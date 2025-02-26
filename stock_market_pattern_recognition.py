import cv2
import torch
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model_path = "trainsmall.pt"  # Path to the trained YOLOv8 model
model = YOLO(model_path)

# Folder to analyze
input_folder = "random_charts"  # Folder containing the images
output_folder = "detections4"  # Folder to save detected images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Variables for collecting information
total_files = 0
detected_files = 0
formation_counts = {}
confidence_values = []

# Process all test images
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png")):
        total_files += 1
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        # Make predictions using the model
        results = model.predict(img, conf=0.75, iou=0.7)  # Lowered IoU threshold

        # Check objects detected by the model
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else []  # Confidence scores
            classes = result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else []  # Detected classes

            # Filter only those with confidence > 75%
            for box, conf, cls in zip(boxes, confidences, classes):
                if conf > 0.75:  # If model confidence is high enough
                    detected_files += 1
                    formation_name = model.names[int(cls)]
                    confidence_values.append(conf)

                    # Update formation count
                    if formation_name in formation_counts:
                        formation_counts[formation_name] += 1
                    else:
                        formation_counts[formation_name] = 1

                    # Center the bounding box and save (as in your previous code)
                    x1, y1, x2, y2 = map(int, box[:4])
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)

                    # Create label and save the image
                    label = f"{formation_name} ({conf:.2f})"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    new_filename = f"{filename.split('.')[0]}_{formation_name}_{int(conf*100)}.jpg"
                    output_path = os.path.join(output_folder, new_filename)
                    cv2.imwrite(output_path, img)

                    print(f"✅ Formation detected: {filename} → {new_filename} (Coordinates: {x1, y1, x2, y2})")

# Generate report
print("\n📊 Report:")
print(f"Total files analyzed: {total_files}")
print(f"Files with formations detected: {detected_files}")
print("Formation types and counts:")
for formation, count in formation_counts.items():
    print(f"{formation}: {count}")

# Plot graphs
plt.figure(figsize=(12, 6))

# Distribution of formation types
plt.subplot(1, 2, 1)
plt.bar(formation_counts.keys(), formation_counts.values(), color='skyblue')
plt.title("Distribution of Formation Types")
plt.xlabel("Formation Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Distribution of confidence values
plt.subplot(1, 2, 2)
plt.hist(confidence_values, bins=20, color='lightgreen', edgecolor='black')
plt.title("Distribution of Confidence Values")
plt.xlabel("Confidence Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
