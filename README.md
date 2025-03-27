# Human-Detection
This project implements real-time human detection using the YOLOv8 segmentation model with OpenCV. It detects people in a webcam feed, highlights them with bounding boxes, and applies a red overlay mask for enhanced visualization.
# Features
* Real-Time Object Detection – Detects humans using YOLOv8.
* Segmentation Mask Overlay – Applies a red mask over detected persons.
* Bounding Box & Confidence Score – Draws bounding boxes with accuracy scores.
* GPU Acceleration – Uses CUDA if available, else runs on CPU.

# Working Process
* Loads YOLOv8 Model: The YOLOv8n-seg model is loaded to detect objects.
* Captures Webcam Feed: Reads real-time video from the default webcam.
* Runs Object Detection: Processes each frame to detect and segment humans.
* Applies Segmentation Mask: Overlays a red mask for detected persons.
* Draws Bounding Boxes: Displays detection confidence scores.
* Displays Output: Shows processed frames with real-time updates.
