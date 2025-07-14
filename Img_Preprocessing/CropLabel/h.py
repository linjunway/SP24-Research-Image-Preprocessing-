import os
import cv2  # OpenCV for image processing
from ultralytics import YOLO

# Load your trained model
model = YOLO('../best.pt')  # Adjust the path as necessary

# Directory containing images for inference
predict_dir = 'predict/'  # Adjust this path as necessary

# Directory to save cropped images
output_dir = 'yolov8_crops/'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Get a list of all files in the prediction directory
files = os.listdir(predict_dir)

# Specify allowed image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add other extensions if needed

# Run inference on each image
for file in files:
    if file.lower().endswith(image_extensions):  # Check if the file is an image
        img_path = os.path.join(predict_dir, file)
        
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        
        # Perform inference
        results = model.predict(source=img_path, conf=0.25)  # Adjust confidence threshold as needed
        
        # Check if results were detected
        for result in results:
            boxes = result.boxes  # Accessing the boxes attribute
            
            # Check if any bounding boxes were detected
            if len(boxes.xyxy) == 0 or boxes.xyxy.shape[0] == 0:
                print(f"No bounding boxes detected in {file}. Skipping.")
                continue  # Skip to the next file if no boxes are detected

            # Iterate over each bounding box
            for i in range(len(boxes.xyxy)):  # Iterate over detected boxes
                xmin, ymin, xmax, ymax = boxes.xyxy[i]  # Access each box
                
                # Convert coordinates to integers
                xmin, ymin, xmax, ymax = map(int, (xmin.item(), ymin.item(), xmax.item(), ymax.item()))

                # Crop the image using the bounding box
                cropped_img = img[ymin:ymax, xmin:xmax]

                # Get label name based on class index (if you have a list of names)
                label_name = result.names[int(boxes.cls[i].item())]  # Access the class label
                
                # Create a filename for the cropped image
                cropped_img_filename = f"{file[:-4]}_crop_{i+1}_{label_name}.jpg"  # Add index and label to the filename
                
                # Save the cropped image to the output directory
                output_path = os.path.join(output_dir, cropped_img_filename)
                cv2.imwrite(output_path, cropped_img)

                print(f"Saved cropped image to: {output_path}")
    else:
        print(f"Skipping non-image file: {file}")
