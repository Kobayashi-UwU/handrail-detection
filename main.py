import cv2
import numpy as np

# Load YOLO model and class labels
net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Set confidence threshold and NMS threshold
confidence_threshold = 0.9
nms_threshold = 0.4

# Open the video file (change the path to your video file)
cap = cv2.VideoCapture('your_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 270))

    # Prepare the frame for YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detected objects' information
    boxes = []
    confidences = []
    class_ids = []

    # Process detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # YOLO returns bounding box coordinates as a fraction of the frame size
                # Convert them to pixel coordinates
                width, height = frame.shape[1], frame.shape[0]
                x = int(detection[0] * width)
                y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate coordinates for drawing bounding box
                x_min = int(x - w / 2)
                y_min = int(y - h / 2)
                x_max = int(x + w / 2)
                y_max = int(y + h / 2)

                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Filter for detections labeled as 'person'
    for i in indices:
        class_id = class_ids[i]
        label = str(classes[class_id])

        if label == 'person':
            box = boxes[i]
            x_min, y_min, x_max, y_max = box
            confidence = confidences[i]

            color = (0, 255, 0)  # BGR color for the bounding box (here, it's green)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Human Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
