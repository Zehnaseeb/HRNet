import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

# Load the TFLite model
tflite_model_path = '/home/shreya/Downloads/model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Prepare input data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 288)),  # (height, width)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the camera
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
orig_h, orig_w = video_capture.get()
output = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 2, (orig_w, orig_h))
# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()
cnt = 0
while video_capture.isOpened():
    # Capture a single frame
    ret, frame = video_capture.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Failed to capture a frame.")
        exit()

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform the same preprocessing as before
    preprocessed_img = transform(image_rgb).unsqueeze(dim=0)
    preprocessed_img = preprocessed_img.numpy().astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Perform further processing with the output data as needed
    print("Output Data:", output_data)

    final_heatmaps = output_data[0]
    for heatmap in final_heatmaps:
        (y,x) = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        h, w = heatmap.shape
        rescaled_x, rescaled_y = int(x*orig_w/w), int(y*orig_h/h)
        cv2.circle(frame, (rescaled_x, rescaled_y), 5, (0,0,255), -1)

    output.write(frame)
    cv2.imshow("frame", frame)
    cnt += 1
    print(cnt)
    # if cnt == 25:
    #     break
    key = cv2.waitKey(1)
    if key == ord("q"):  # press 'q' on keyboard to quit the screen
        break

# Release the camera
video_capture.release()
output.release()