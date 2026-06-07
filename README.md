# Face-Detection-with-Haar-Cascades

# NAME:NANDIKA S

# REG NO:212224230175

# EXP NO:12

## Aim

To write a Python program using OpenCV and Matplotlib to perform the following operations: Extract Region of Interest (ROI) from an image. Perform face detection using Haar Cascade classifiers on static images. Perform eye detection in images. Perform face detection with labels in real-time webcam video. Display the processed results using Matplotlib. 

## Software Required
-Anaconda - Python 3.7 or above
-OpenCV library (opencv-python)
-Matplotlib library (matplotlib)
-Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

Step 1: Import the required libraries such as cv2, numpy, matplotlib.pyplot, and os.

Step 2: Load the input image using cv2.imread() and convert it into RGB and grayscale formats using cv2.cvtColor().

Step 3: Display the original image using plt.imshow().

Step 4: Extract the Region of Interest (ROI) using image slicing and create the segmented image using masking techniques.

Step 5: Apply Gaussian Blur to the grayscale image and perform Canny Edge Detection using cv2.Canny().

Step 6: Detect contours from the edge image using cv2.findContours() and draw bounding rectangles around significant contours.

Step 7: Load the MobileNet-SSD model files, create a blob from the image, and perform object detection using the deep learning network.

Step 8: Display all processed outputs such as segmented ROI, edge detection, contour detection, and object detection results using Matplotlib

## Program

# I) ROI Segmentation in an Image using Bitwise AND

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Read the image and convert the image into RGB
image = cv2.imread('Qno. 1.jpg')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()
# Step 4: Set the pixels to display the ROI (Region of Interest)
# Define the coordinates for the Region of Interest (ROI)
# (startY:endY, startX:endX)
roi = image[100:420, 200:550]  # ROI coordinates (adjust as needed)

# Create a blank mask of the same size as the original image
mask = np.zeros_like(image)

# Place the ROI on the mask
mask[100:420, 200:550] = roi
# Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
segmented_roi = cv2.bitwise_and(image, mask)
# Step 6: Display the segmented ROI from the image
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()

```

# II) Handwriting Detection in an Image

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Read the image and convert it to RGB for displaying
image = cv2.imread('your_image_1.jpg')  # Replace with your actual image file path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# Original Image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
# Step 3: Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur (5x5 kernel)
# Step 5: Use Canny edge detector to find edges
edges = cv2.Canny(blurred_image, 50, 150)  # Detect edges using Canny (thresholds 50 and 150)
# Canny Edge Detection
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
# Step 6: Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Step 7: Filter contours based on area and draw bounding boxes
result_image = image.copy()  # Create a copy of the original image to draw bounding boxes
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter out small areas
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box for the contour
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle
# Handwriting Detection Result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')

```

# III) Object Detection with Labels in an Image using MobileNet-SSD

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
config_file = 'deploy.prototxt'  # Path to the config file
weights = 'mobilenet_iter_73000.caffemodel'  # Path to the weights file
# Step 2: Use a pretrained DNN model (MobileNet-SSD v3)
net = cv2.dnn.readNetFromCaffe(config_file, weights)
# Step 4: Create a class label and print the same
class_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
                18: 'sofa', 19: 'train', 20: 'tvmonitor'}
# Step 5: Read the image
image = cv2.imread('download.webp')  # Replace with your image path
(h, w) = image.shape[:2]
# Convert image to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Create a blob for DNN processing
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
# Step 6: Set the model and threshold to 0.5
net.setInput(blob)
detections = net.forward()
# Step 7: Flatten the index, confidence
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # Confidence threshold
        index = int(detections[0, 0, i, 1])  # Get class index
        label = class_labels[index]  # Get label name
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
# Step 8: Draw rectangles and labels on the image
        cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# Step 9: Display the image using Matplotlib
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()

```
## OUTPUT
<img width="964" height="487" alt="image" src="https://github.com/user-attachments/assets/7842dbb1-7154-486d-bae0-76c4ee377efe" />

<img width="745" height="465" alt="image" src="https://github.com/user-attachments/assets/bc4cbbdd-ac74-41cc-9085-0ab6f3e50862" />

<img width="429" height="545" alt="image" src="https://github.com/user-attachments/assets/08a0efd6-3af8-4fc1-8de0-f26169026747" />

<img width="391" height="584" alt="image" src="https://github.com/user-attachments/assets/96c208a7-f7b1-43d6-8cde-154c530e68c6" />

<img width="402" height="577" alt="image" src="https://github.com/user-attachments/assets/96d70092-a860-4d94-bba9-add5aecf83ca" />

<img width="721" height="504" alt="image" src="https://github.com/user-attachments/assets/b3c05340-041a-420b-910f-e828520daf85" />

## Result
Thus, face and eye detection were successfully implemented using Haar Cascades in Python. The program detected faces and eyes in static images and performed labeled real-time face detection from webcam feed using OpenCV
