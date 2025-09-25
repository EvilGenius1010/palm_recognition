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


    # Apply CLAHE for local contrast enhancement 
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply Unsharp Masking to sharpen lines 
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 20.0)
    sharpened = cv2.addWeighted(enhanced, 2.0, blurred, -0.75, 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(sharpened, threshold1=80, threshold2=150)

    # Save or return preprocessed image
    cv2.imwrite("preprocessed_edges.jpg", edges)
    return edges

# Example usage
preprocessed_image = preprocess_palm_image("data/sample1.jpg")