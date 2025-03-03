import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the names of all the layers in the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read input image
img = cv2.imread("C:/Users/DELL/Desktop/Atul/PyProject/ObjectDetection/objectDetection_env/img/pen.jpg")
height, width, channels = img.shape

# Convert the image to a blob (a format that YOLO understands)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the blob as input to the network
net.setInput(blob)

# Run forward pass (get the output from the network)
outs = net.forward(output_layers)

# Initialize lists to hold detected bounding boxes and confidence scores
class_ids = []
confidences = []
bboxes = []

# Process the detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Only consider detections with confidence > 0.5
        if confidence > 0.5:
            # Get the center coordinates, width, and height of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Get the top-left coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Append the class ID, confidence score, and bounding box coordinates
            class_ids.append(class_id)
            confidences.append(float(confidence))
            bboxes.append([x, y, w, h])

# Apply Non-Maximum Suppression (NMS) to remove duplicate detections
indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

# Draw the bounding boxes and class labels on the image
for i in indices.flatten():
    x, y, w, h = bboxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    # Draw the rectangle and label
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detected objects
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
