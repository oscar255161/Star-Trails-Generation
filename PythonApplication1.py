import os
import cv2
import sys
sys.setrecursionlimit(100000)
import numpy as np
from multiprocessing import Pool
import imageio
import tkinter as tk
from imageio import get_writer
from PIL import Image,ImageSequence, ImageTk
from tkinter import filedialog

def sobel_filter(img):
    # Apply Sobel filter to compute gradients in x and y directions
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Combine x and y gradient results using NumPy operations for efficiency
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient = np.uint8(np.clip(gradient, 0, 255))
    return gradient

# ===================mopology================================
def close_image(img, erode_kernel_size=3, dilate_kernel_size=3):
    # Use OpenCV morphology functions for closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, dilate_kernel_size))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

# ===================edge_segmentation================================
def dft_highpass_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # HPF
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30 
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shifted * mask

    # IDFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # normali
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back

def blur_Canny(img):
    highpassed_img = dft_highpass_filter(img)
    edges = cv2.Canny(highpassed_img, 100, 200)
    return edges

# ===================GIF================================
def create_star_trail_gif_v2(original_img, output_path, center, image_name, x_line, star_clusters, total_frames, angle_step , duration=20 , combine_frame=10):
    height, width = original_img.shape[:2]
    gif_images = []

    #  create mask
    star_draw_mask = np.ones_like(original_img, dtype=np.uint8) * 255
    for x in range(x_line.shape[1]):
        y_coords = np.where(x_line[:, x] == 0)[0]
        if y_coords.size > 0:
            y_min = np.min(y_coords)
            star_draw_mask[y_min:, x] = 0

    # Precompute rotated positions for all stars
    angle_rads = np.deg2rad(np.arange(total_frames) * angle_step)
    rotated_positions = np.array([
        (
            center[0] + (cluster['location'][0] - center[0]) * np.cos(angle_rads) - (cluster['location'][1] - center[1]) * np.sin(angle_rads),
            center[1] + (cluster['location'][0] - center[0]) * np.sin(angle_rads) + (cluster['location'][1] - center[1]) * np.cos(angle_rads)
        )
        for cluster in star_clusters
    ])

    # Create an image to accumulate star trails for PNG
    accumulated_stars_png = np.zeros_like(original_img)

    # Create an image to accumulate star trails for GIF
    accumulated_stars_gif = np.zeros_like(original_img)

    # Process every N frames at a time
    for i in range(0, total_frames, combine_frame):
        frame_img = np.zeros_like(original_img)
        # Draw stars for the current N frames
        for j in range(combine_frame):
            if i + j < total_frames:
                for idx, cluster in enumerate(star_clusters):
                    x_rotated, y_rotated = rotated_positions[idx, 0, i + j], rotated_positions[idx, 1, i + j]
                    brightness = cluster['brightness']  # Same brightness within the 5-frame unit
                    area = cluster['area']
                    radius = max(1, min(3, int(area / 10)))

                    if 0 <= x_rotated < width and 0 <= y_rotated < height and np.all(star_draw_mask[int(y_rotated), int(x_rotated)] == 255):
                        cv2.circle(frame_img, (int(x_rotated), int(y_rotated)), radius, (brightness, brightness, brightness), -1, lineType=cv2.LINE_AA)

        # Accumulate the stars for GIF and PNG
        accumulated_stars_png = cv2.add(accumulated_stars_png, frame_img)
        accumulated_stars_gif = cv2.add(accumulated_stars_gif, frame_img)

        # Add the accumulated image to gif_images every 5 frames
        combined_img = cv2.add(original_img, accumulated_stars_gif)
        gif_images.append(Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)))

        # Apply brightness decay for the next set of 5 frames in the GIF
        decay_factor = 0.98  # Adjust as needed for the desired decay rate
        accumulated_stars_gif = (accumulated_stars_gif * decay_factor).astype(np.uint8)

    # Save the final star trail image as PNG using accumulated_stars_png
    star_trail_img = cv2.add(original_img, accumulated_stars_png)
    pil_img_png = Image.fromarray(cv2.cvtColor(star_trail_img, cv2.COLOR_BGR2RGB))
    output_img_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}_star_trail.png")
    pil_img_png.save(output_img_path)

    # Save the GIF animation
    output_gif_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}_star_trail.gif")
    gif_images[0].save(output_gif_path, save_all=True, append_images=gif_images[1:], duration=duration , loop=0)

# ================= ==find_brightest_star================================
def process_contour(args):
    img_gray, contour, x_line_y_coords = args
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_gray, mask=mask)

    if max_loc[1] < x_line_y_coords[max_loc[0]]:
        return max_val, max_loc, cv2.contourArea(contour)
    else:
        return None

def find_brightest_star(img, x_line_img, star_threshold, max_contours=150):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Vectorized approach to find y coordinates of the white line in x_line_img
    if np.all(x_line_img == 255):
        x_line_y_coords = np.full(img_gray.shape[1], img_gray.shape[0])
    else:
        x_line_y_coords = np.argmax(x_line_img[::-1, :] > 0, axis=0)
        x_line_y_coords = img_gray.shape[0] - x_line_y_coords - 1
        x_line_y_coords[x_line_y_coords == img_gray.shape[0] - 1] = 0

    # Apply threshold and find contours
    _, threshold_img = cv2.threshold(img_gray, star_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # prepare for paralle parameter
    args = [(img_gray, contour, x_line_y_coords) for contour in contours]

    # process
    with Pool() as pool:
        results = pool.map(process_contour, args)

    
    processed_contours = [res for res in results if res is not None]

    
    if len(processed_contours) > max_contours:
        processed_contours = sorted(processed_contours, key=lambda x: x[0], reverse=True)[:max_contours]

   
    star_clusters = [{'brightness': brightness, 'location': location, 'area': area} 
                     for brightness, location, area in processed_contours]

    # find brightest
    if star_clusters:
        overall_brightest_point = max(star_clusters, key=lambda x: x['brightness'])
    else:
        overall_brightest_point = {'brightness': 0, 'location': (0, 0), 'area': 0}

    return star_clusters, overall_brightest_point['location'], overall_brightest_point['brightness'], overall_brightest_point['area']

# ================= ==other================================
def del_sky_v3(img):

   
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # init_setting
    one_seventh_height = img.shape[0] / 7
    one_half_height = img.shape[0] / 3
    half_width = img.shape[1] / 2
    total_area = img.shape[0] * img.shape[1]
    min_area_threshold = total_area / 500

    
    x_min_y = {}
    a = 0

    for label in range(1, num_labels):
        component_area = stats[label, cv2.CC_STAT_AREA]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        component_top = stats[label, cv2.CC_STAT_TOP]

        
        if component_top < one_seventh_height and component_width < half_width:
            continue
        len_limit=30
        if (component_area >= min_area_threshold or
            (component_width > img.shape[1] / len_limit and component_height > img.shape[0] / len_limit)):
                a = 1
                component_mask = (labels == label)
                ys, xs = np.where(component_mask)
                for x, y in zip(xs, ys):
                    if x not in x_min_y or y < x_min_y[x]:
                        x_min_y[x] = y

    
    if a == 0:
        img = np.ones_like(img, dtype=np.uint8) * 255
    else:
       
        img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for x, y in x_min_y.items():
            img[y, x] = 255  

    return img

def connect_line_v4(img):
    height, width = img.shape[:2]
    
    # Create a fully white image
    connected_img = np.full((height, width), 255, dtype=np.uint8)

    if np.all(img == 255):
        return img
    else:
        # Find white pixels (non-zero values) in the grayscale image
        white_pixels = np.where(img > 0)
    
        # Extract the x and y coordinates of white pixels
        white_x, white_y = white_pixels[1], white_pixels[0]
    
        # Find the leftmost and rightmost white pixels
        leftmost_x = np.min(white_x)
        rightmost_x = np.max(white_x)
    
        
        y_for_leftmost_x = white_y[np.argmin(white_x)]
        y_for_rightmost_x = white_y[np.argmax(white_x)]
    
        # Draw horizontal lines from the leftmost and rightmost white pixels
        if leftmost_x != 0:
            cv2.line(connected_img, (0, y_for_leftmost_x), (leftmost_x, y_for_leftmost_x), (0, 0, 0), 1)
    
        if rightmost_x != (width - 1):
            cv2.line(connected_img, (rightmost_x, y_for_rightmost_x), (width - 1, y_for_rightmost_x), (0, 0, 0), 1)
    
        # Sort the white pixels by their x-coordinate
        sorted_indices = np.argsort(white_x)
        sorted_white_x = white_x[sorted_indices]
        sorted_white_y = white_y[sorted_indices]

        # Iterate over the sorted white pixels and draw lines
        for i in range(len(sorted_white_x) - 1):
            x1, y1 = sorted_white_x[i], sorted_white_y[i]
            x2, y2 = sorted_white_x[i + 1], sorted_white_y[i + 1]
            cv2.line(connected_img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Fill below the connected line with black color
        for x in range(width):
            y_coords = np.where(connected_img[:, x] == 0)[0]
            if y_coords.size > 0:
                y_min = np.min(y_coords)
                connected_img[y_min:, x] = 0

        return connected_img
class Application:
    def __init__(self, root, output_path):
        self.root = root
        self.output_path = output_path
        self.root.title("星軌圖")
    
    
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, pady=10)  

    
        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5) 

        self.auto_button = tk.Button(button_frame, text="Auto Find Star", command=self.auto_find_star)
        self.auto_button.pack(side=tk.LEFT, padx=5)  
        
        self.manual_button = tk.Button(button_frame, text="Manual Select Star", command=self.manual_select_star)
        self.manual_button.pack(side=tk.LEFT, padx=5)  

   
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill="both", expand=True)

        self.selected_point = None
 
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            original_img = cv2.imread(file_path)
            #original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            self.original_img = original_img  
            self.image_name = os.path.basename(file_path) 
        
       
            scaled_img = self.scale_image_for_display(original_img)
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(scaled_img))
            
            self.display_image(scaled_img)

    def scale_image_for_display(self, img):
    
        self.canvas.update()  
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

    
        img_height, img_width = img.shape[:2]
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        self.scale = min(scale_width, scale_height)

 
        scaled_width = int(img_width * self.scale)
        scaled_height = int(img_height * self.scale)
        return cv2.resize(img, (scaled_width, scaled_height))

    def display_image(self, scaled_img):

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = scaled_img.shape[1], scaled_img.shape[0]
        x_position = (canvas_width - img_width) // 2
        y_position = (canvas_height - img_height) // 2

     
        self.canvas.delete("all") 
        self.canvas.create_image(x_position, y_position, image=self.photo, anchor=tk.NW)

    def auto_find_star(self):
        if self.original_img is not None:
            processed_img = sobel_filter(self.original_img)
            processed_img = blur_Canny(processed_img)
            processed_img = close_image(processed_img)
            del_skyy = del_sky_v3(processed_img)
            x_line_img = connect_line_v4(del_skyy)
            height, width = self.original_img.shape[:2]
            if width * height < 1000000:
                star_threshold=140
                total_frames=720
                angle_step=0.5
                duration=25
                combine_frame=20
            elif 1000000<=width * height < 2000000:
                star_threshold=160
                total_frames=720
                angle_step=0.5
                duration=20
                combine_frame=30
            elif 2000000<=width * height < 3000000:
                star_threshold=180
                total_frames=720
                angle_step=0.5
                duration=10
                combine_frame=40
            elif 3000000<=width * height < 5000000:
                star_threshold=190
                total_frames=720
                angle_step=0.5
                duration=5
                combine_frame=60
            elif 5000000<=width * height<10000000:
                star_threshold=200
                total_frames=360
                angle_step=1
                duration=5
                combine_frame=80
            elif width * height>=10000000:
                star_threshold=200
                total_frames=360
                angle_step=1
                duration=5
                combine_frame=120
            star_clusters, (brightest_x, brightest_y), _, _ = find_brightest_star(self.original_img, x_line_img, star_threshold)
            self.create_star_trail_gif((brightest_x, brightest_y), x_line_img, star_clusters)
            self.root.destroy() 
    def manual_select_star(self):
        self.canvas.bind("<ButtonPress-1>", self.on_click)

    def on_click(self, event):
  
        canvas_x, canvas_y = event.x, event.y

 
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.photo.width(), self.photo.height()
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2
 
 
        if hasattr(self, 'scale') and self.scale != 0:
            original_x = int((canvas_x - x_offset) / self.scale)
            original_y = int((canvas_y - y_offset) / self.scale)
            if 0 <= original_x < self.original_img.shape[1] and 0 <= original_y < self.original_img.shape[0]:
                self.selected_point = (original_x, original_y)
                print(f"Selected point in original image: {self.selected_point}")

                if self.original_img is not None:
         
                    processed_img = sobel_filter(self.original_img)
                    processed_img = blur_Canny(processed_img)
                    processed_img = close_image(processed_img)
                    del_skyy = del_sky_v3(processed_img)
                    x_line_img = connect_line_v4(del_skyy)
                    height, width = self.original_img.shape[:2]
                    if width * height < 1000000:
                        star_threshold=140
                        total_frames=720
                        angle_step=0.5
                        duration=25
                        combine_frame=20
                    elif 1000000<=width * height < 2000000:
                        star_threshold=160
                        total_frames=720
                        angle_step=0.5
                        duration=20
                        combine_frame=30
                    elif 2000000<=width * height < 3000000:
                        star_threshold=180
                        total_frames=720
                        angle_step=0.5
                        duration=10
                        combine_frame=40
                    elif 3000000<=width * height < 5000000:
                        star_threshold=190
                        total_frames=720
                        angle_step=0.5
                        duration=5
                        combine_frame=60
                    elif 5000000<=width * height<10000000:
                        star_threshold=200
                        total_frames=360
                        angle_step=1
                        duration=5
                        combine_frame=80
                    elif width * height>=10000000:
                        star_threshold=200
                        total_frames=360
                        angle_step=1
                        duration=5
                        combine_frame=120
                    star_clusters, _, _, _ = find_brightest_star(self.original_img, x_line_img, star_threshold)

        
                    self.create_star_trail_gif((original_x, original_y), x_line_img, star_clusters)
          
                    self.root.destroy() 
            else:
                print("Clicked outside the image area.")
        else:
            print("Scale factor not set or invalid.")



    def create_star_trail_gif(self, center, x_line, star_clusters):
   
        if hasattr(self, 'original_img') and self.original_img is not None and hasattr(self, 'image_name'):
                height, width = self.original_img.shape[:2]
                if width * height < 1000000:
                    star_threshold=140
                    total_frames=720
                    angle_step=0.5
                    duration=25
                    combine_frame=20
                elif 1000000<=width * height < 2000000:
                    star_threshold=160
                    total_frames=720
                    angle_step=0.5
                    duration=20
                    combine_frame=30
                elif 2000000<=width * height < 3000000:
                    star_threshold=180
                    total_frames=720
                    angle_step=0.5
                    duration=10
                    combine_frame=40
                elif 3000000<=width * height < 5000000:
                    star_threshold=190
                    total_frames=720
                    angle_step=0.5
                    duration=5
                    combine_frame=60
                elif 5000000<=width * height<10000000:
                    star_threshold=200
                    total_frames=360
                    angle_step=1
                    duration=5
                    combine_frame=80
                elif width * height>=10000000:
                    star_threshold=200
                    total_frames=360
                    angle_step=1
                    duration=5
                    combine_frame=120
                create_star_trail_gif_v2(self.original_img, self.output_path, center, self.image_name, x_line, star_clusters,total_frames,angle_step , duration , combine_frame)
        else:
            print("Error: No image loaded or image name not set.")
             
def main():
    input_folder = os.path.join(os.getcwd(), 'input')
    output_folder = os.path.join(os.getcwd(), 'output')
    root = tk.Tk()
    app = Application(root, output_folder)
    root.mainloop()

if __name__ == "__main__":
    main()
