import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter import *
import customtkinter as ctk
import sys

from ctypes import windll

from PIL import Image, ImageTk
import fitz  # install PyMuPDF
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from ultralytics import YOLO

# initialize global variables
stop_flag = False
default_flag = True
ml_flag = False

uploaded_files = []
sel_clear_button = None
save_location = None
progressbar = None

"""================================================================================================================================================================="""
""" Shortcut Helper Functions                                                                                                                                       """
"""================================================================================================================================================================="""

def extract_first_page_as_image(pdf_path, image_path, dpi=100):
    pdf_file = fitz.open(pdf_path)
    page = pdf_file.load_page(0)    # Read the first page
    pix = page.get_pixmap(dpi=dpi)  # Render page to an image
    pix.save(image_path)
    return Image.open(image_path)

def convert_to_grayscale(image_np):
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image_np, ksize=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image_np, ksize, sigmaX)

def detect_edges(image_np):
    return cv2.Canny(image_np, 50, 200, apertureSize=3)

def adaptive_detect_edges(image_np, low_threshold = 50, high_threshold = 150):    # Perform Canny edge detection with adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        image_np, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    return cv2.Canny(adaptive_thresh, low_threshold, high_threshold)

def adjust_exposure(image_np, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)

def enhance_contrast(image): # creates issue where parts of the image when cropped are blacked out
    """
    Enhance the contrast of the input image using CLAHE.
    
    :param image: Input image as a NumPy array
    :return: Image with enhanced contrast
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
    else:
        # Color image (BGR)
        # Convert the image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge the enhanced L channel back with A and B channels
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        
        # Convert the LAB image back to BGR color space
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def apply_bilateral_filter(image_np, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)

def apply_morphological_operations(image_np):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)

"""=============================================================================================================================================================="""
""" Background Removal Helper Functions                                                                                                                          """
"""=============================================================================================================================================================="""

"""
def detect_background_color(image, k=2):
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Apply K-means clustering to find the dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Find the largest cluster (the most dominant color)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = dominant_colors[np.argmax(counts)]
    
    return dominant_color

def change_background_to_transparent(image, tolerance=30, k=2):
    if image is None:
        print("Invalid image array")
        return None

    # Check the number of channels in the image
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Convert single-channel image to 3 channels (grayscale to BGR)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert the image to RGBA if it is not already
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Detect the background color using K-means clustering
    background_color = detect_background_color(image[:, :, :3], k)
    print(f"Detected background color: {background_color}")

    # Define the lower and upper bounds for the background color
    lower_bound = np.array([max(0, c - tolerance) for c in background_color])
    upper_bound = np.array([min(255, c + tolerance) for c in background_color])

    # Create a mask where the background color is detected
    mask = cv2.inRange(image[:, :, :3], lower_bound, upper_bound)

    # Expand the mask to account for small differences in lighting
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Set the alpha channel to 0 where the mask is white
    image[:, :, 3] = np.where(mask == 255, 0, image[:, :, 3])

    return image
"""

"""================================================================================================================================================================="""
""" Contour Detection Helper Functions                                                                                                                              """
"""================================================================================================================================================================="""

def find_contours(edges, min_area=100, edge_threshold=50):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = edges.shape

    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (cv2.contourArea(cnt) >= min_area and 
            x > edge_threshold and y > edge_threshold and 
            x + w < width - edge_threshold and 
            y + h < height - edge_threshold):
            filtered_contours.append(cnt)
    
    return filtered_contours

def is_fully_within(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)

def filter_contours(contours):
    contours.sort(key=cv2.contourArea, reverse=True)
    filtered_contours = []
    for i, cnt1 in enumerate(contours):
        keep = True
        for j, cnt2 in enumerate(contours):
            if i != j and is_fully_within(cnt1, cnt2):
                keep = False
                break
        if keep:
            filtered_contours.append(cnt1)
    return filtered_contours

def are_contours_close(contour1, contour2, width_threshold=100, height_threshold=100):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    return not (x1 > x2 + w2 + width_threshold or x1 + w1 < x2 - width_threshold or y1 > y2 + h2 + height_threshold or y1 + h1 < y2 - height_threshold)

def connect_contours(image_np, contours, width_threshold=100, height_threshold=100):
    merged_contours = []
    while contours:
        base_contour = contours.pop(0)
        connected_contours = [base_contour]
        i = 0
        while i < len(contours):
            if are_contours_close(base_contour, contours[i], width_threshold, height_threshold):
                connected_contours.append(contours.pop(i))
            else:
                i += 1
        
        base_contour = np.vstack(connected_contours)
        merged_contours.append(base_contour)
    
    # Convert merged contours to convex hulls to ensure enclosure
    hulls = [cv2.convexHull(c) for c in merged_contours]
    return hulls

def connect_overlapping_contours(image_np, contours):
    merged_contours = []
    while contours:
        base_contour = contours.pop(0)
        connected_contours = [base_contour]
        i = 0
        while i < len(contours):
            if are_contours_close(base_contour, contours[i], 0, 0):  # Distance threshold set to 0 for overlapping
                connected_contours.append(contours.pop(i))
            else:
                i += 1
        
        base_contour = np.vstack(connected_contours)
        merged_contours.append(base_contour)
    
    # Convert merged contours to convex hulls to ensure enclosure
    hulls = [cv2.convexHull(c) for c in merged_contours]
    return hulls

def remove_small_contours(contours, min_area=100):
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

def extend_bounding_boxes(contours, extension=10):
    """
    Extend the bounding boxes around contours by a specified number of pixels.
    
    :param contours: List of contours
    :param extension: Number of pixels to extend the bounding boxes by
    :return: List of extended bounding boxes (x, y, w, h)
    """
    extended_boxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extend the bounding box by the specified number of pixels
        x_new = max(x - extension, 0)
        y_new = max(y - extension, 0)
        w_new = w + 2 * extension
        h_new = h + 2 * extension
        
        extended_boxes.append((x_new, y_new, w_new, h_new))
    
    return extended_boxes

def draw_extended_boxes(image, extended_boxes):
    """
    Draw extended bounding boxes on the image.
    
    :param image: Input image as a NumPy array
    :param extended_boxes: List of extended bounding boxes (x, y, w, h)
    :return: Image with extended bounding boxes drawn
    """
    height, width = image.shape[:2]

    for (x, y, w, h) in extended_boxes:
        # Ensure the bounding boxes do not go beyond the image boundaries
        x_end = min(x + w, width - 1)
        y_end = min(y + h, height - 1)
        cv2.rectangle(image, (x, y), (x_end, y_end), (0, 255, 0), 2)
    
    return image

def draw_contours(image_np, contours):
    image_with_contours = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image_with_contours, [box], 0, (0, 0, 255), 2)
    return image_with_contours

def save_contour_coordinates(contours, output_path):
    with open(output_path, 'w') as f:
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ymin_xmin = tuple(box[0])
            ymax_xmax = tuple(box[2])
            ymin_xmax = tuple(box[1])
            ymax_xmin = tuple(box[3])
            f.write(f'{ymin_xmin}, {ymax_xmax}, {ymin_xmax}, {ymax_xmin}\n')


"""================================================================================================================================================================="""
""" Image Translation Helper Functions                                                                                                                              """
"""================================================================================================================================================================="""

def calculate_rotation_angle(rect):
    angle = rect[2]
    if rect[1][0] < rect[1][1]:  # If the width is smaller than height
        angle = -(angle + 90) if angle < -45 else -angle
    return angle

def rotate_image(image_np, center, angle):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def crop_rotated_image(rotated_image, rotated_box):
    x, y, w, h = cv2.boundingRect(rotated_box)
    return Image.fromarray(rotated_image).crop((x, y, x + w, y + h))

"""================================================================================================================================================================="""
""" Main Code                                                                                                                                                       """
"""================================================================================================================================================================="""
def process_image(input_image, save, alpha=1.0, beta=0, min_area=100, edge_threshold=50, width_threshold=100, height_threshold=100):
    if stop_flag:
        return

    # Convert the PIL image to a NumPy array
    image_np = np.array(input_image)  

    #rm_bg_image = change_background_to_transparent(blurred_image_np, tolerance=30, k=2)
    #cv2.imwrite("output_images/bg_removed.png", rm_bg_image)

    # Convert to grayscale
    grayscale_image_np = convert_to_grayscale(image_np)

    # Apply Gaussian blur
    blurred_image_np = apply_gaussian_blur(grayscale_image_np) 
    
    # Adjust exposure
    image_exp_np = adjust_exposure(blurred_image_np, alpha, beta)
    #cv2.imwrite("output_images/adj_exposure.png", image_exp_np)                                         # TESTING

    # Enhance contrast
    enhanced_image_np = cv2.equalizeHist(image_exp_np)
    #cv2.imwrite("output_images/adj_contrast.png", enhanced_image_np)                                    # TESTING

    # Apply bilateral filter
    filtered_image_np = apply_bilateral_filter(enhanced_image_np)
    #cv2.imwrite("output_images/adj_bilateral_filter.png", filtered_image_np)                            # TESTING

    # Apply morphological operations
    morphed_image_np = apply_morphological_operations(filtered_image_np)
    #cv2.imwrite("output_images/morphed.png", morphed_image_np)                                          # TESTING

    # Use OpenCV to detect edges
    edges = adaptive_detect_edges(morphed_image_np, low_threshold = 50, high_threshold = 150)
    #cv2.imwrite("output_images/edges_detected.png", edges)                                              # TESTING

    # Find contours in the edged image
    contours = find_contours(edges, min_area, edge_threshold)

    # Filter contours to remove smaller fully overlapping ones
    filtered_contours = filter_contours(contours)

    # Connect and merge close contours
    connected_contours = connect_contours(image_np, filtered_contours, width_threshold, height_threshold)

    # Connect and merge overlapping contours
    final_contours = connect_overlapping_contours(image_np, connected_contours)

    # Perform another round of connecting close contours with a different distance threshold
    final_contours = connect_contours(image_np, final_contours, width_threshold * 0.5, height_threshold * 0.5)  # For example, reducing the distance threshold

    # Connect and merge overlapping contours again
    final_contours = connect_overlapping_contours(image_np, final_contours)

    # Perform a third round of connecting close contours with another different distance threshold
    final_contours = connect_contours(image_np, final_contours, width_threshold * 0.25, height_threshold * 0.25)  # Further reducing the distance threshold

    # Connect and merge overlapping contours one more time
    final_contours = connect_overlapping_contours(image_np, final_contours)

    # Perform a fourth round of connecting close contours with another different distance threshold
    final_contours = connect_contours(image_np, final_contours, width_threshold * 0.1, height_threshold * 0.1)  # Further reducing the distance threshold

    # Connect and merge overlapping contours one last time
    final_contours = connect_overlapping_contours(image_np, final_contours)
 
    # Remove small contours
    final_contours = remove_small_contours(final_contours, 250)

    #extended_contours = extend_bounding_boxes(final_contours, extension=10)
    #image_with_contours = draw_extended_boxes(morphed_image_np, extended_contours)
    #cv2.imwrite("output_images/detected_contours_with_boxes.png", image_with_contours)  # Save the image with the detected contours and bounding boxes

    # Rotate and crop each detected contour to horizontal position
    for idx, cnt in enumerate(final_contours):
        rect = cv2.minAreaRect(cnt)
        # print(rect)
        angle = calculate_rotation_angle(rect)
        #print(angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        center = (rect[0][0], rect[0][1])
        
        rotated_image_np = rotate_image(image_np, center, angle)
        rotated_box = cv2.transform(np.array([box]), cv2.getRotationMatrix2D(center, angle, 1.0))[0]
        # print(rotated_box)
        
        cropped_image = crop_rotated_image(rotated_image_np, rotated_box)

        # Save the cropped image in the selected save location
        # Construct the file path using the save_location and the index
        save_path = os.path.join(save, f"cropped_contour_{idx}.png")
        
        # Save the image to the specified directory
        cropped_image.save(save_path)

def ml_process (uploaded_files, save_location):
    if stop_flag:
        return
    if getattr(sys, 'frozen', False):
        model_path = os.path.join(sys._MEIPASS, 'best.pt')
    else:
        model_path = 'path/to/best.pt'  # Path when running in a script
    # print(f"Loading model from: {model_path}")

    model = YOLO(model_path)  # Adjust the path as necessary
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Run inference on each image
    for file in uploaded_files:
        if file.lower().endswith(image_extensions):  # Check if the file is an image
            # Read the image using OpenCV
            img = cv2.imread(file)
                
            # Perform inference
            results = model.predict(source=file, conf=0.25)  # Adjust confidence threshold as needed
                
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
                    cropped_img_filename = f"{os.path.basename(file[:-4])}_crop_{i+1}_{label_name}.jpg"  # Add index and label to the filename
                        
                    # Save the cropped image to the output directory
                    output_path = os.path.join(save_location, cropped_img_filename)
                    cv2.imwrite(output_path, cropped_img)

                    print(f"Saved cropped image to: {output_path}")
        else:
            print(f"Skipping non-image file: {file}")
    return

"""================================================================================================================================================================="""
""" GUI Commands Code                                                                                                                                               """
"""================================================================================================================================================================="""

def upload_file():
    global file_label, count_label, file_path, uploaded_files, sel_clear_button

    file_path = fd.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.pdf")])
    remove_existing_labels()

    if sel_clear_button:
        sel_clear_button.destroy()
        
    sel_clear_button = ctk.CTkButton(selector_frame, text="Clear File(s)", command=clear_uploaded)
    sel_clear_button.pack(side=BOTTOM, anchor="center", pady=(0,10))

    if file_path:
        uploaded_files.append(file_path)
        file_label = ctk.CTkLabel(selector_frame, text=f"File uploaded: {file_path}")
        file_label.pack()
        count_label = ctk.CTkLabel(selector_frame, text="Number of files uploaded: 1")
        count_label.pack()

        #uploaded_files = os.path.abspath(uploaded_files)
    else:
        messagebox.showwarning("No File", "No file was selected.")

def upload_files():
    global file_label, count_label, file_paths, uploaded_files, sel_clear_button
    file_paths = fd.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.pdf")])
    remove_existing_labels()

    if sel_clear_button:
        sel_clear_button.destroy()
    
    sel_clear_button = ctk.CTkButton(selector_frame, text="Clear File(s)", command=clear_uploaded)
    sel_clear_button.pack(side=BOTTOM, anchor="center", pady=(0,10))

    if file_paths:
        num_files = len(file_paths)
        for file in file_paths:
            uploaded_files.append(file)
        file_label = ctk.CTkLabel(selector_frame, text=f"Files uploaded:\n" + "\n".join(file_paths))
        file_label.pack()
        count_label = ctk.CTkLabel(selector_frame, text=f"Number of files uploaded: {num_files}")
        count_label.pack()
        
        #uploaded_files = os.path.abspath(uploaded_files)
    else:
        messagebox.showwarning("No File", "No files were selected.")

def remove_existing_labels():
    global file_label, count_label
    if file_label:
        file_label.destroy()
        file_label = None
    if count_label:
        count_label.destroy()
        count_label = None

def update_upload_button():
    global upload_button
    # Remove the existing upload button if it exists
    if upload_button:
        upload_button.destroy()

    # Create a new upload button based on the selected mode
    if radio_var.get() == "single":
        upload_button = ctk.CTkButton(selector_frame, text="Upload File", command=upload_file)
    else:
        upload_button = ctk.CTkButton(selector_frame, text="Upload Files", command=upload_files)

    upload_button.pack(anchor="center", pady=(5,0))

def choose_save_location():
    global save_location, sl
    save_location = fd.askdirectory()

    if sl:
        sl.destroy()

    if save_location:
        sl = ctk.CTkLabel(save_frame, text=f"{save_location}")
        sl.pack()
    else:
        messagebox.showwarning("No Save Location", "No save location was selected.")

def default_processing():
    global default_flag
    global ml_flag
    default_flag = True
    ml_flag = False

def ml_processing():
    global default_flag
    global ml_flag
    ml_flag = True
    default_flag = False

def force_stop():
    global stop_flag
    stop_flag = True
    progressbar.destroy()
    messagebox.showinfo("Functions Stopped", "All functions have been stopped.")

def reset_stop_flag():
    global stop_flag
    stop_flag = False

def run_function():
    reset_stop_flag

    global progressbar, save_location, uploaded_files

    if not uploaded_files:
        messagebox.showerror("Error", "No files selected!")
        return

    if not save_location:
        messagebox.showerror("Error", "No save location selected!")
        return

    if progressbar:
        progressbar.destroy()

    progressbar = ctk.CTkProgressBar(exec_frame, mode="determinate")
    progressbar.pack(side=LEFT, padx=20)
    progressbar.start()

    if default_flag:
        for image_path in uploaded_files:
            image = Image.open(image_path)
            process_image(image, save_location)

            # Save processed image
        messagebox.showinfo("Processing Complete", f"File saved at: {save_location}")
        progressbar.destroy()
    else:
        ml_process(uploaded_files, save_location)
        messagebox.showinfo("Processing Complete", f"File saved at: {save_location}")
        progressbar.destroy()

def clear_uploaded():
    global uploaded_files, file_label, count_label, sel_clear_button
    uploaded_files = []

    if file_label:
        file_label.destroy()

    if count_label:
        count_label.destroy()

    if sel_clear_button:
        sel_clear_button.destroy()

    messagebox.showinfo("Files Cleared", "All uploaded files have been cleared.")

"""================================================================================================================================================================="""
""" GUI Code                                                                                                                                                        """
"""================================================================================================================================================================="""

# adjust resolution of text
windll.shcore.SetProcessDpiAwareness(1)

# create main window
root = tk.Tk()
root.resizable(0,0)
root.title("Piction - Image Preprocessing")

ctk.set_default_color_theme("blue")
ctk.set_appearance_mode("system")

# add icon image
path = "./Zhejiang_University_Logo.svg.png"
load = Image.open(path)
render = ImageTk.PhotoImage(load)
root.iconphoto(False, render)

# base frame
window = tk.Frame(root, padx=10, pady=10, bg="gray")
window.pack(fill="both", expand=True)

# introduction frame
intro_frame = ctk.CTkFrame(window)
intro_frame.grid(row=0, column=0, rowspan=3, columnspan=5, padx=5, pady=(5,10), ipadx=10, ipady=5, sticky="nsew",)

ctk.CTkLabel(intro_frame, text="Welcome to Piction!").pack()
ctk.CTkLabel(intro_frame, text="Piction is a Python-based program which automates the detection, extraction, and rotation of the objects found in given image(s).").pack()
ctk.CTkLabel(intro_frame, text="Simply choose the files you want to be processed, select a save location, and run.").pack()

# file mode selector frame
file_label = None
count_label = None

file_path = None
file_paths = None

selector_frame = ctk.CTkFrame(window)
selector_frame.grid(row=3, column=0, rowspan=3, columnspan=5, padx=5, pady=(0,10), ipadx=5, ipady=10, sticky="nsew")

ctk.CTkLabel(selector_frame, text="Select file option").pack(anchor="w", padx=10)
radio_var = tk.StringVar(value="single")

single = ctk.CTkRadioButton(selector_frame, radiobutton_height=16, radiobutton_width=16, text="Single File", variable=radio_var, value="single", command=update_upload_button)
single.pack(anchor="w", padx=20)

multi = ctk.CTkRadioButton(selector_frame, radiobutton_height=16, radiobutton_width=16, text="Multiple Files", variable=radio_var, value="multiple", command=update_upload_button)
multi.pack(anchor="w", padx=20)

upload_button = None
update_upload_button()

# choosing save location frame
sl = None

save_frame = ctk.CTkFrame(window)
save_frame.grid(row=6, column=0, rowspan=3, columnspan=5, padx=5, pady=(0,10), ipadx=5, ipady=10, sticky="nsew")

ctk.CTkLabel(save_frame, text="Select Save Location").pack(anchor="w", padx=10)
save_loc_button = ctk.CTkButton(save_frame, text="Choose Save Location", command=choose_save_location)
save_loc_button.pack(anchor="center", padx=20, pady=5)

# processing mode frame
processing_mode = None

mode_frame = ctk.CTkFrame(window)
mode_frame.grid(row=9, column=0, rowspan=3, columnspan=5, padx=5, pady=(0,10), ipadx=5, ipady=10, sticky="nsew")

ctk.CTkLabel(mode_frame, text="Choose Preferred Mode of Processing").pack(anchor="w", padx=10)
processing_var = tk.StringVar(value="mode1")

default = ctk.CTkRadioButton(mode_frame, radiobutton_height=16, radiobutton_width=16, text="Default (In testing)", variable=processing_var, value="mode1", command=default_processing)
default.pack(anchor="w", padx=20)

ml = ctk.CTkRadioButton(mode_frame, radiobutton_height=16, radiobutton_width=16, text="Machine Learning", variable=processing_var, value="mode2", command=ml_processing)
ml.pack(anchor="w", padx=20)

# execution frame
progressbar = None

exec_frame = ctk.CTkFrame(window)
exec_frame.grid(row=12, column=0, rowspan=1, columnspan=5, padx=5, pady=(0,5), ipadx=5, ipady=10, sticky="nsew")

run_button = ctk.CTkButton(exec_frame, text="Run", command=run_function)
run_button.pack(side=RIGHT, padx=(0,5))
#run_button.grid(pady=10, sticky="se")

cancel_button = ctk.CTkButton(exec_frame, text="Cancel", command=force_stop)
cancel_button.pack(side=RIGHT, padx=5)
#cancel_button.grid(padx=5, pady=10, sticky="se")

"""================================================================================================================================================================="""
""" Program Execution Code                                                                                                                                          """
"""================================================================================================================================================================="""

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    root.mainloop()