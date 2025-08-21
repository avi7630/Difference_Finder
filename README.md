# Difference_Finder
This script is a desktop application that uses python and OpenCV to automatically align and compare two images, highlighting differences for quality control or identifying modifications.

This project was born as a way to find defects in a PCB using Image Proccesing, therefor, it is present in some forms in the code

Co-authored-by: Name <DanielKudarsky@users.noreply.github.com>


# Image Difference Finder

This is a Python-based desktop application for finding and visualizing differences between two images. It is built using `OpenCV` for powerful image processing and `Tkinter` for a user-friendly graphical interface.

## Features

-   **Image Alignment**: Automatically aligns images using `ORB` (Oriented FAST and Rotated BRIEF) feature detection and `homography` to ensure accurate comparisons, even if the images are slightly misaligned or taken from different angles.
-   **Difference Detection**: Compares images using a combination of structural (grayscale) and optional color difference analysis.
-   **Preprocessing**: Includes an optional color correction feature (`equalize_image`) to normalize brightness and contrast, which helps in identifying genuine content differences instead of variations caused by lighting conditions.
-   **Interactive GUI**: Provides a simple interface to load images and run the comparison.
-   **Detailed Output**: Displays a side-by-side view of the full images with differences highlighted. It also generates a scrollable gallery of "zoomed-in" views for each detected difference, making it easy to inspect details.

## How to Run

### Prerequisites

You need to have Python installed on your system. This application also requires the following libraries:

-   `opencv-python`
-   `numpy`
-   `Pillow` (PIL)

You can install them using `pip`:

```bash
pip install opencv-python numpy Pillow cv2
```

# Execution
Clone this repository or download the find file [here](https://github.com/avi7630/Difference_Finder/blob/main/find%20diff.py) .

Open your terminal or command prompt.

Navigate to the directory where you saved the file.

Run the script after making sure the required libraries are installed

# How to Use
The application window will open and you will be promoting by 
![this window](https://github.com/avi7630/Difference_Finder/blob/main/Screenshot%202025-08-21%20164916.png)



Click the "Browse" button next to "Original Image Path" to select your first image.

Click the "Browse" button next to "Modified Image Path" to select the second image you want to compare.

Optionally, check or uncheck the "Apply Color Correction" and "Detect Color Differences" boxes.

Click the "Run Comparison" button.
(you may expect different results if the paths are switched)

Watch the results

![](https://github.com/avi7630/Difference_Finder/blob/main/Screenshot%202025-08-21%20165527.png)
![](https://github.com/avi7630/Difference_Finder/blob/main/Screenshot%202025-08-21%20165553.png)
Code Structure
The script is divided into two main parts:

Image Processing Functions: A set of functions at the top that handle the core logic, including align_images, equalize_image, and zoom_problems_side_by_side.

GUI Application Class: The ImageComparisonApp class, which uses Tkinter to manage the user interface, handle file selection, and display the final results.
