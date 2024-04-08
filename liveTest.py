import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('modelYolo.keras')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Adjust the camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
className = ['Thumbs Up', 'Thumbs Downs', 'Fist', 'Index']
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if unable to capture a frame

    # Preprocess the frame (resize, scale, etc.) according to your model's requirements
    # Here it's assumed that your model expects 640x640 RGB images
    input_frame = cv2.resize(frame, (320, 320))
    input_frame = input_frame  # Normalize to [0,1] if your model expects this range
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(input_frame)

    # Extract boxes, confidences, and classes from the model's output
    boxes = predictions['boxes'][0]  # Assuming the first element is what we want
    confidences = predictions['confidence'][0]
    classes = predictions['classes'][0]

    # Iterate over detections
    for i, box in enumerate(boxes):
        if i > 2:
            continue
        confidence = confidences[i]
        class_id = classes[i]

        # Filter out weak detections by ensuring the 'confidence' is greater than a minimum threshold
        if confidence > 0.1:  # You can adjust this threshold
            # Scale box to original image size
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            label = f"Class: {className[class_id]}, Confidence: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
