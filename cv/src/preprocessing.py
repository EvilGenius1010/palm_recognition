import cv2
import numpy as np
from PIL import Image

def preprocess_palm_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)


    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.convertScaleAbs(sobel_x + sobel_y)


    # Morphological operations to clean up edges
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(sobel_edges, kernel, iterations=1)
    edges = cv2.erode(sobel_edges, kernel, iterations=1)


    # Save or return preprocessed image
    cv2.imwrite("preprocessed_edges.jpg", edges)
    return edges

# Example usage
preprocessed_image = preprocess_palm_image("data/sample1.jpg")