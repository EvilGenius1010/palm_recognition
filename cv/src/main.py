# import torch
# from torchvision import models, transforms
# from PIL import Image # imports python image library
# import sys # access to command line arguments
# from torchvision.models.feature_extraction import create_feature_extractor

# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# # Load EfficientNet-B0 with latest pretrained weights
# weights = EfficientNet_B0_Weights.DEFAULT
# model = efficientnet_b0(weights=weights)

# # model.eval()

# # Preprocessing pipeline
# transform = weights.transforms()


# def run_inference(image_path):
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)
        
#     # Define which layer to extract from
#     return_nodes = {
#         'features.7': 'feature_map'  # Last convolutional block
#     }

#     # Create extractor
#     feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
#     feature_extractor.eval()

#     # Run image through extractor
#     features = feature_extractor(input_tensor)['feature_map']
#     print("Extracted feature shape:", features.shape)

#     with torch.no_grad():
#         output = model(input_tensor)
#         prediction = torch.argmax(output, dim=1).item()
#         print(f"Predicted class index: {prediction}")

# if __name__ == "__main__":
#     run_inference(sys.argv[1])


import os
from preprocessing import preprocess_palm_image
from segmentation import segment_hand
from main import run_inference  
import cv2

def process_dataset(input_dir, preprocessed_dir, segmented_dir):
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_dir, fname)

        # Step 1: Preprocessing
        preprocessed_path = os.path.join(preprocessed_dir, fname)
        preprocessed_image = preprocess_palm_image(image_path)
        cv2.imwrite(preprocessed_path, preprocessed_image)

        # Step 2: Segmentation
        segmented_path = os.path.join(segmented_dir, fname)
        segmented_image = segment_hand(preprocessed_path)
        cv2.imwrite(segmented_path, segmented_image)

        # Step 3: Inference
        run_inference(segmented_path)

# Example usage
if __name__ == "__main__":
    process_dataset(
        input_dir="dataset/images",
        preprocessed_dir="dataset/preprocessed_images",
        segmented_dir="dataset/seg_images"
    )
