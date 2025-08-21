import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Scrollbar, Canvas, messagebox
from PIL import Image, ImageTk, ImageEnhance

# ----------------------------------------------------
# Image Processing Functions
# ----------------------------------------------------

def equalize_image(image):
    """
    Applies white balance and contrast equalization to an image.
    Uses LAB color space to avoid affecting colors while enhancing brightness.
    """
    # Convert to LAB color space for white balance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    white_balanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Convert to LAB and apply CLAHE for contrast
    lab = cv2.cvtColor(white_balanced_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    final_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return final_image

def align_images(image_to_align, reference_image):
    """
    Aligns two images using the ORB feature detector and homography.
    This is crucial for ensuring pixel-by-pixel comparison is accurate.
    """
    img1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography to warp the image
    height, width, _ = reference_image.shape
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))
    return aligned_image

def zoom_problems_side_by_side(original_img, modified_img, contours_img):
    """
    Creates pairs of zoomed-in images showing the detected defects.
    It crops a region around each defect for easy inspection.
    """
    imgs_pair = []
    for i in range(len(contours_img)):
        x, y, w, h = cv2.boundingRect(contours_img[i])
        
        # Define a buffer zone around the defect
        buffer = 50
        
        # Ensure the cropped area stays within image boundaries
        x_start = max(x - buffer, 0)
        y_start = max(y - buffer, 0)
        x_end = min(x + w + buffer, original_img.shape[1])
        y_end = min(y + h + buffer, original_img.shape[0])

        # Adjust crop box to fit in image and maintain aspect ratio
        if x - buffer < 0:
            x_end = min(x_end + (buffer - x), original_img.shape[1])
        if y - buffer < 0:
            y_end = min(y_end + (buffer - y), original_img.shape[0])
        if x + w + buffer > original_img.shape[1]:
            x_start = max(x_start - ((x + w + buffer) - original_img.shape[1]), 0)
        if y + h + buffer > original_img.shape[0]:
            y_start = max(y_start - ((y + h + buffer) - original_img.shape[0]), 0)

        original_defect = original_img[y_start:y_end, x_start:x_end].copy()
        modified_defect = modified_img[y_start:y_end, x_start:x_end].copy()
        
        # Draw a red rectangle around the defect area
        cv2.rectangle(original_defect, (x - x_start, y - y_start), (x - x_start + w, y - y_start + h), (0, 0, 255), 2)
        cv2.rectangle(modified_defect, (x - x_start, y - y_start), (x - x_start + w, y - y_start + h), (0, 0, 255), 2)
        
        imgs_pair.append((original_defect, modified_defect))
    return imgs_pair

# ----------------------------------------------------
# GUI Application Class
# ----------------------------------------------------

class ImageComparisonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Find Image Differences")
        self.geometry("600x280")
        self.original_path = ""
        self.modified_path = ""
        self.create_widgets()

    def create_widgets(self):
        """
        Sets up the main GUI window with buttons, labels, and checkboxes.
        """
        input_frame = tk.Frame(self)
        input_frame.pack(pady=10)
        
        # Original image selection
        self.label1 = Label(input_frame, text="Original Image Path:")
        self.label1.grid(row=0, column=0, padx=5, pady=5)
        self.entry1 = Label(input_frame, text="", bg="white", relief="sunken", width=50)
        self.entry1.grid(row=0, column=1, padx=5, pady=5)
        self.button1 = tk.Button(input_frame, text="Browse", command=lambda: self.select_image("original"))
        self.button1.grid(row=0, column=2, padx=5, pady=5)
        
        # Modified image selection
        self.label2 = Label(input_frame, text="Modified Image Path:")
        self.label2.grid(row=1, column=0, padx=5, pady=5)
        self.entry2 = Label(input_frame, text="", bg="white", relief="sunken", width=50)
        self.entry2.grid(row=1, column=1, padx=5, pady=5)
        self.button2 = tk.Button(input_frame, text="Browse", command=lambda: self.select_image("modified"))
        self.button2.grid(row=1, column=2, padx=5, pady=5)
        
        # Checkbox for color correction
        self.color_correction_var = tk.BooleanVar(value=True) # Default is checked
        color_correction_checkbox = tk.Checkbutton(input_frame, text="Apply Color Correction", variable=self.color_correction_var)
        color_correction_checkbox.grid(row=2, column=0, columnspan=3, pady=5)

        # Message label for color correction
        self.color_correction_msg_var = tk.StringVar()
        self.color_correction_msg_label = Label(input_frame, textvariable=self.color_correction_msg_var, fg="blue", font=("Arial", 10))
        self.color_correction_msg_label.grid(row=3, column=0, columnspan=3, pady=(0, 5))
        self.color_correction_msg_var.set("The color correction is being used for bringing the altered image closer to the original one.")

        # New checkbox for color difference detection
        self.color_detection_var = tk.BooleanVar(value=True) # Default is checked
        color_detection_checkbox = tk.Checkbutton(input_frame, text="Detect Color Differences", variable=self.color_detection_var)
        color_detection_checkbox.grid(row=4, column=0, columnspan=3, pady=5)

        # Message label for color difference detection
        self.color_detection_msg_var = tk.StringVar()
        self.color_detection_msg_label = Label(input_frame, textvariable=self.color_detection_msg_var, fg="blue", font=("Arial", 10))
        self.color_detection_msg_label.grid(row=5, column=0, columnspan=3, pady=(0, 5))
        self.color_detection_msg_var.set("the color differences is being used for finding more changes and might detect a false positive.")

        # Button to run the comparison
        confirm_button = tk.Button(self, text="Run Comparison", command=self.run_comparison)
        confirm_button.pack(pady=20)

    def select_image(self, type):
        """
        Opens a file dialog to select an image file.
        """
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            if type == "original":
                self.original_path = path
                self.entry1.config(text=path)
            elif type == "modified":
                self.modified_path = path
                self.entry2.config(text=path)

    def run_comparison(self):
        """
        Main function to perform the image comparison logic.
        """
        if not (self.original_path and self.modified_path):
            messagebox.showerror("Error", "Please select both images.")
            return
        
        original = cv2.imread(self.modified_path)
        image_mod = cv2.imread(self.original_path)
        
        if original is None:
            messagebox.showerror("Error", f"Could not load original image from: {self.original_path}. Please check the file path.")
            return
        if image_mod is None:
            messagebox.showerror("Error", f"Could not load modified image from: {self.modified_path}. Please check the file path.")
            return
        
        # Apply color correction based on checkbox state
        if self.color_correction_var.get():
            final_original = equalize_image(original)
            final_image_mod = equalize_image(image_mod)
        else:
            final_original = original
            final_image_mod = image_mod

        rows, _, _ = final_original.shape
        aligned_image_mod = align_images(final_image_mod, final_original)

        # Grayscale difference (structural changes)
        original_gray = cv2.cvtColor(final_original, cv2.COLOR_BGR2GRAY)
        modified_gray = cv2.cvtColor(aligned_image_mod, cv2.COLOR_BGR2GRAY)
        blurred_original = cv2.GaussianBlur(original_gray, (5, 5), 0)
        blurred_modified = cv2.GaussianBlur(modified_gray, (5, 5), 0)
        diff_orig_mod = cv2.absdiff(blurred_original, blurred_modified)
        _, threshold_diff_mod = cv2.threshold(diff_orig_mod, 100, 255, cv2.THRESH_BINARY)
        
        final_threshold_mask = threshold_diff_mod.copy()

        # New: Color difference detection
        if self.color_detection_var.get():
            # Apply a stronger blur to HSV images to focus on large color differences
            blurred_hsv_original = cv2.GaussianBlur(final_original, (55, 55), 0)
            blurred_hsv_modified = cv2.GaussianBlur(aligned_image_mod, (55, 55), 0)

            # Convert blurred images to HSV
            hsv_original = cv2.cvtColor(blurred_hsv_original, cv2.COLOR_BGR2HSV)
            hsv_modified = cv2.cvtColor(blurred_hsv_modified, cv2.COLOR_BGR2HSV)
            
            # Split channels
            h_orig, _, _ = cv2.split(hsv_original)
            h_mod, _, _ = cv2.split(hsv_modified)

            # Calculate difference for Hue channel only
            diff_h = cv2.absdiff(h_orig, h_mod)

            # Threshold the hue difference within a specific range
            lower_bound = np.array([45])
            upper_bound = np.array([55])
            threshold_h = cv2.inRange(diff_h, lower_bound, upper_bound)
            
            # Combine structural and color differences
            final_threshold_mask = cv2.bitwise_or(final_threshold_mask, threshold_h)
            
        # Apply morphological operations to close gaps and merge areas
        kernel = np.ones((int(rows/70), int(rows/70)), np.uint8)
        final_threshold_mask = cv2.dilate(final_threshold_mask, kernel, iterations=1)
        
        # Find contours of defects
        contours_mod, _ = cv2.findContours(final_threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to remove small noise
        min_defect_area = 50 # Adjust this value to filter out smaller noise
        valid_contours = []
        for contour in contours_mod:
            area = cv2.contourArea(contour)
            if area > min_defect_area:
                valid_contours.append(contour)
        
        # Create zoomed-in views of the defects
        enlarged_defects_pair = zoom_problems_side_by_side(final_original, aligned_image_mod, valid_contours)
        
        # Mark the defects on the full images
        marked_original = final_original.copy()
        marked_modified = aligned_image_mod.copy()
        cv2.drawContours(marked_original, valid_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(marked_modified, valid_contours, -1, (0, 0, 255), 2)
        
        self.show_results_window(marked_original, marked_modified, valid_contours, enlarged_defects_pair)

    def show_results_window(self, marked_original, marked_modified, contours, enlarged_defects_pair):
        """
        Creates and displays a new window with the comparison results.
        Includes a side-by-side view of the full images and zoomed-in views of defects.
        """
        
        results_window = Toplevel(self)
        results_window.title("Comparison Results")
        results_window.state('zoomed')  # Also for Windows, ensures maximized
        
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        main_image_view_frame = tk.Frame(main_frame)
        main_image_view_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        main_title = Label(main_image_view_frame, text="Full Image Comparison", font=("Arial", 14, "bold"))
        main_title.pack(pady=(0, 10))
        
        # Original Image Frame
        main_orig_frame = tk.Frame(main_image_view_frame)
        main_orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        orig_canvas = Canvas(main_orig_frame, highlightthickness=0)
        orig_canvas.pack(fill="both", expand=True)
        self.orig_image_pil = Image.fromarray(cv2.cvtColor(marked_original, cv2.COLOR_BGR2RGB))
        self.orig_img_tk = ImageTk.PhotoImage(image=self.orig_image_pil)
        self.orig_image_on_canvas = orig_canvas.create_image(0, 0, anchor="nw", image=self.orig_img_tk)
        self.orig_canvas = orig_canvas
        self.orig_image_pil_full = self.orig_image_pil.copy()

        # Modified Image Frame
        main_mod_frame = tk.Frame(main_image_view_frame)
        main_mod_frame.pack(side="left", fill="both", expand=True)
        mod_canvas = Canvas(main_mod_frame, highlightthickness=0)
        mod_canvas.pack(fill="both", expand=True)
        self.mod_image_pil = Image.fromarray(cv2.cvtColor(marked_modified, cv2.COLOR_BGR2RGB))
        self.mod_img_tk = ImageTk.PhotoImage(image=self.mod_image_pil)
        self.mod_image_on_canvas = mod_canvas.create_image(0, 0, anchor="nw", image=self.mod_img_tk)
        self.mod_canvas = mod_canvas
        self.mod_image_pil_full = self.mod_image_pil.copy()

        # Add image labels below the canvases
        Label(main_orig_frame, text="Second Image", font=("Arial", 12)).pack(pady=(5, 0))
        Label(main_mod_frame, text="Original Image", font=("Arial", 12)).pack(pady=(5, 0))

        # Configure canvases to be scrollable
        orig_canvas.config(scrollregion=orig_canvas.bbox("all"))
        mod_canvas.config(scrollregion=mod_canvas.bbox("all"))

        # Shared state for synchronized zoom and pan
        self.zoom_level = 1.0
        self.last_x, self.last_y = 0, 0
        self.pan_start_x, self.pan_start_y = 0, 0
        
        # Bind events to both canvases
        self.orig_canvas.bind("<MouseWheel>", self.zoom)
        self.mod_canvas.bind("<MouseWheel>", self.zoom)
        self.orig_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.mod_canvas.bind("<ButtonPress-1>", self.start_pan)
        self.orig_canvas.bind("<B1-Motion>", self.pan)
        self.mod_canvas.bind("<B1-Motion>", self.pan)
        
        separator = tk.Frame(main_frame, width=2, bg='gray', relief='sunken')
        separator.pack(side="left", fill="y", padx=5)
        
        defect_frame = tk.Frame(main_frame)
        defect_frame.pack(side="right", fill="y", expand=True, padx=(5, 0))
        
        defect_title = Label(defect_frame, text="Enlarged Differences", font=("Arial", 14, "bold"))
        defect_title.pack(pady=(0, 10))
        scrollbar = Scrollbar(defect_frame)
        scrollbar.pack(side="right", fill="y")
        canvas = Canvas(defect_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Display the zoomed-in defect pairs
        for i, (orig_defect, mod_defect) in enumerate(enlarged_defects_pair):
            defect_pair_frame = tk.Frame(scrollable_frame, bd=1, relief="solid")
            defect_pair_frame.pack(pady=5, padx=5, fill="x")
            orig_defect_rgb = cv2.cvtColor(orig_defect, cv2.COLOR_BGR2RGB)
            img_pil_orig_defect = Image.fromarray(orig_defect_rgb)
            img_pil_orig_defect.thumbnail((150, 150))
            img_tk_orig_defect = ImageTk.PhotoImage(image=img_pil_orig_defect)
            orig_defect_label = Label(defect_pair_frame, image=img_tk_orig_defect, text="Original", compound="top")
            orig_defect_label.image = img_tk_orig_defect
            orig_defect_label.pack(side="left", padx=5)
            mod_defect_rgb = cv2.cvtColor(mod_defect, cv2.COLOR_BGR2RGB)
            img_pil_mod_defect = Image.fromarray(mod_defect_rgb)
            img_pil_mod_defect.thumbnail((150, 150))
            img_tk_mod_defect = ImageTk.PhotoImage(image=img_pil_mod_defect)
            mod_defect_label = Label(defect_pair_frame, image=img_tk_mod_defect, text=f"Difference {i+1}", compound="top")
            mod_defect_label.image = img_tk_mod_defect
            mod_defect_label.pack(side="left", padx=5)
            
        count_label = Label(results_window, text=f"Total differences found: {len(enlarged_defects_pair)}")
        count_label.pack(pady=10)

    def zoom(self, event):
        # Determine the scale factor
        scale_factor = 1.1 if event.delta > 0 else 1.0 / 1.1
        
        # Update the zoom level
        self.zoom_level *= scale_factor
        
        # Limit the zoom to prevent issues
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        if self.zoom_level > 5.0: self.zoom_level = 5.0

        # Calculate new image sizes based on the full-resolution copies
        new_size_orig = (int(self.orig_image_pil_full.width * self.zoom_level),
                         int(self.orig_image_pil_full.height * self.zoom_level))
        new_size_mod = (int(self.mod_image_pil_full.width * self.zoom_level),
                        int(self.mod_image_pil_full.height * self.zoom_level))

        # Resize images and update PhotoImage
        resized_orig = self.orig_image_pil_full.resize(new_size_orig, Image.LANCZOS)
        self.orig_img_tk = ImageTk.PhotoImage(resized_orig)
        self.orig_canvas.itemconfig(self.orig_image_on_canvas, image=self.orig_img_tk)

        resized_mod = self.mod_image_pil_full.resize(new_size_mod, Image.LANCZOS)
        self.mod_img_tk = ImageTk.PhotoImage(resized_mod)
        self.mod_canvas.itemconfig(self.mod_image_on_canvas, image=self.mod_img_tk)

        # Update the scroll region to match the new image size
        self.orig_canvas.configure(scrollregion=self.orig_canvas.bbox("all"))
        self.mod_canvas.configure(scrollregion=self.mod_canvas.bbox("all"))

        # Implement zoom at cursor by adjusting the canvas view
        view_orig_x = self.orig_canvas.canvasx(event.x)
        view_orig_y = self.orig_canvas.canvasy(event.y)
        self.orig_canvas.xview_scroll(int(view_orig_x * (1 - scale_factor)), "units")
        self.orig_canvas.yview_scroll(int(view_orig_y * (1 - scale_factor)), "units")

        view_mod_x = self.mod_canvas.canvasx(event.x)
        view_mod_y = self.mod_canvas.canvasy(event.y)
        self.mod_canvas.xview_scroll(int(view_mod_x * (1 - scale_factor)), "units")
        self.mod_canvas.yview_scroll(int(view_mod_y * (1 - scale_factor)), "units")


    def start_pan(self, event):
        self.orig_canvas.scan_mark(event.x, event.y)
        self.mod_canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.orig_canvas.scan_dragto(event.x, event.y, gain=1)
        self.mod_canvas.scan_dragto(event.x, event.y, gain=1)
        
if __name__ == "__main__":
    app = ImageComparisonApp()
    app.mainloop()


