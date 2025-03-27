from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Select the device: GPU (if available) or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 model on the frame
    results = model(frame, conf=0.4, imgsz=640, verbose=False)

    # Process results
    for result in results:
        if result.masks is None:
            continue  # Skip if no masks detected

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, cls in enumerate(classes):
            if int(cls) == 0:  # Class 0 corresponds to "Person"
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = confidences[i]

                # Resize mask to match the frame size
                mask = cv2.resize(masks[i], (frame.shape[1], frame.shape[0]))

                # Create an overlay with a transparent mask
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[:, :, 2] = 255  # Red color mask
                frame[mask > 0.5] = cv2.addWeighted(frame[mask > 0.5], 0.5, colored_mask[mask > 0.5], 0.5, 0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display confidence label
                label = f"Human {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the real-time frame
    cv2.imshow("Real-Time Human Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
