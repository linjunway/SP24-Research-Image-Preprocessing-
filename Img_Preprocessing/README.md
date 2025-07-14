# Piction - Image Preprocessor with GUI and Object Detection

**Piction** is a GUI-based Python application designed for **large-scale image preprocessing** and **object extraction**. It is particularly useful in workflows where multiple objects (e.g., scanned diagrams, text blocks, or shapes) must be isolated and saved from a single large image.

The application supports both:
- **Classic image processing techniques** (edge detection, contour analysis, contrast enhancement, etc.): less accurate when used on more complex images where the  object does not significantly contrast the background.
- **Machine learning-based object detection** (using a pretrained YOLO model): greater accuracy overall.

## Features

- Select and upload single or multiple images or PDFs.
- Choose between default image processing or YOLO-based object detection.
- Auto-detect and remove image background clutter.
- Identify, connect, and filter relevant contours in images.
- Rotate and crop detected objects into aligned, individual images.
- Save processed outputs in a user-defined folder.
- Supports standard image types: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, and `.pdf`.

## Installer & Executable

This source code of this project can be compiled into a PyInstaller-based setup script to bundle everything into a standalone Windows executable for ease of distribution and usage without Python installation. 

**Note**: The current installer includes all dependencies and the YOLO model, making it **too large for GitHub uploads**. Optimization is ongoing.

---

## Libraries Used

| Library        | Purpose |
|----------------|---------|
| `tkinter`, `customtkinter` | GUI framework |
| `PIL` (Pillow) | Image handling |
| `cv2` (OpenCV) | Image processing |
| `numpy`        | Array manipulation |
| `scikit-learn` | KMeans clustering (background detection) |
| `PyMuPDF (fitz)` | PDF to image conversion |
| `ultralytics`  | YOLO object detection |
| `multiprocessing` | Freeze support for Windows executable |
| `os`, `sys`    | Filesystem and system control |

---

## How It Works

1. **User Interface**: Launches a friendly GUI to guide the user through:
   - Uploading image(s)
   - Selecting a save folder
   - Choosing between default or ML-based processing

2. **Default Processing**:
   - Applies grayscale, blur, histogram equalization
   - Detects and merges contours
   - Crops and saves objects in the image based on edge-based detection

3. **ML (YOLO) Processing**:
   - Loads a pretrained YOLO model (`best.pt`)
   - Detects bounding boxes from the image
   - Crops and saves detected objects based on labels and confidence

---

## Future Improvements

- **Reduce Installer Size**:
  - Bundle only required Python dependencies
  - Use dynamic model loading instead of embedding the YOLO weights
  - Create separate download links for large assets such as the model file `best.pt`

- **Improve Model Accuracy**:
  - Allow model fine-tuning from within the app (future training interface)
  - Add feedback loop for user-labeled corrections

- **UI/UX Enhancements**:
  - Improve layout scaling and responsiveness
  - Add progress indicators for long image batches
  - Drag-and-drop support for files

- **Export Options**:
  - Export metadata like detected bounding boxes, labels, and angles
  - Save as multiple formats (`.png`, `.jpg`, `.svg`)
