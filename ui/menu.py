import sys
from time import time
import tkinter as tk
from turtle import width
import cv2
import queue
import time

from tkinter import filedialog
from PIL import Image, ImageTk
from sympy import false
from darknet import darknet
from queue import Queue
from threading import Thread

# Global flag to control the execution of threads
is_running = True
detection_images_thread = None
drawing_thread = None
drawing_ocr_thread = None
show_frame_processed_thread = None

def load_video():

    global cap
    global is_running
    
    video_path = filedialog.askopenfilename(filetypes = [("all video format", ".avi") ])
    cap = cv2.VideoCapture(video_path)
    is_running = True
    
    if(not detection_images_thread.is_alive()):
        detection_images_thread.start()

    if(not drawing_thread.is_alive()):
        drawing_thread.start()

    if(not drawing_ocr_thread.is_alive()):
        drawing_ocr_thread.start()

    if(not show_frame_processed_thread.is_alive()):
        show_frame_processed_thread.start()

    play_video()

def play_video():

    ret, frame = cap.read()  # Read a frame from the video source

    if ret is not None and frame is not None:
        frame_realtime_queue.put(frame)
        resize_and_save_in_queue(frame)
        videoStreamingBox.after(30, play_video)
            
    if(not ret):
        print("www")
        stop_video()

def resize_and_save_in_queue(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to the dimensions expected by Darknet
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)

    # Put the original frame in the frame queue    
    frame_queue.put(frame)

    # Create a Darknet image from the resized frame
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)

    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    # Put the Darknet image in the Darknet image queue    
    darknet_image_queue.put(img_for_detect)

def detections_process(darknet_image_queue, detections_queue):

    while is_running:
        if not darknet_image_queue.empty():
            try:
                darknet_image = darknet_image_queue.get()  # Retrieve an image from the queue
                                        
                # Perform object detection on the image
                detections = darknet.detect_image(network, class_names, darknet_image, thresh=.75)
                
                detections_queue.put(detections)  # Put the detections in the detections queue
                
                darknet.free_image(darknet_image)  # Free the memory of the Darknet image          
                            
            except queue.Empty:
                continue
        else:
            time.sleep(0.01)

def create_crops(frame_queue, detections_queue):
    
    while is_running:
        if not detections_queue.empty():
            try:
                frame = frame_queue.get()  # Retrieve a frame from the queue
                detections = detections_queue.get()  # Retrieve detections for the frame

                detections_adjusted = []
                            
                if frame is not None and frame.size > 0:
                    # Adjust each detection to the original frame size and add to list
                    for label, confidence, bbox in detections:
                        bbox_adjusted = darknet.convert2original(frame, bbox)
                        detections_adjusted.append((str(label), confidence, bbox_adjusted))
                        
                    crops_resized = darknet.get_crops(detections_adjusted, frame)

                    ocr_items = (crops_resized, frame, detections_adjusted)
                    
                    plate_ocr_queue.put(ocr_items)
                    
            except queue.Empty:
                continue
        else:
            time.sleep(0.01)

def drawing_boxes_and_lincese_plate(plate_ocr_queue):

    while is_running:
        if not plate_ocr_queue.empty():
            try:
                cropsAndBbox, frame, detections_adjusted = plate_ocr_queue.get()
                detection_count = 0
                
                for crop, bbox_adjusted in cropsAndBbox:
                    height, width = crop.shape[:2]
                    if crop is not None and crop.size > 0 and  width > 0 and height > 0:
                        left, top, right, bottom = darknet.bbox2points(bbox_adjusted)
                        frame = darknet.read_lincese_plate_by_ocr(frame, crop, detections_adjusted[detection_count], left, top)
                        detection_count += 1

                # Put the processed frame in the queue
                frame_processed_queue.put(frame)
            except queue.Empty:
                continue
        else:
            time.sleep(0.01)

def show_frame_processed():
    
    while is_running:
        try:
            
            prev_time = time.time()  # Record the time before detection
            
            frame = frame_realtime_queue.get()
            if not frame_processed_queue.empty():
                frame = frame_processed_queue.get()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)
            img.thumbnail((1200, 800)) 
            img_tk = ImageTk.PhotoImage(image=img)
            
            videoStreamingBox.create_image(0, 0, anchor="nw", image=img_tk)
            videoStreamingBox.image = img_tk
        
            # Calculate FPS
            fps = int(1 / (time.time() - prev_time))
            #print("FPS: {}".format(fps))  # Print the FPS
                        
            fpsBox.config(text="FPS: {}".format(fps), foreground="red")
               
        except queue.Empty:
            continue

def stop_video():
        
    global is_running
    global videoStreamingBox
    global fpsBox
          
    import cv2
    
    cap.release()    
    videoStreamingBox.delete("all")
    videoStreamingBox.image = None
    fpsBox.configure(text='')
    is_running = False
    cv2.destroyAllWindows()

def create_main_gui():

    splash_root.destroy()
    
    menu = tk.Tk()

    tk.Wm.wm_title(menu, "Main menu")
    menu.minsize(1200, 800)

    frame_main = tk.Frame(menu)
    frame_main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    create_video_canvas(frame_main)
    
    create_buttons(frame_main)
    
    center_on_screen(menu)

    menu.mainloop()
    
def create_video_canvas(frame_main):        
    
    global videoStreamingBox
    global fpsBox
    
    videoStreamingBox = tk.Canvas(frame_main, width=1200, height=800, bg="#c3c1c1")
    videoStreamingBox.grid(row=0, column=0, columnspan=2)
    
    fpsBox = tk.Label(frame_main, justify="left", width=20, height=2)
    fpsBox.grid(row=2, column=0, columnspan=2, pady=10, sticky="w")
    
def create_buttons(frame_main):
    
    button_frame = tk.Frame(frame_main)
    button_frame.grid(row=3, column=0, columnspan=2, pady=20)

    btnStreamingVideo = tk.Button(button_frame, text="Select a video", width=15, command=load_video)
    btnStreamingVideo.grid(row=0, column=0, padx=10)

    btnStop = tk.Button(button_frame, text="Stop", width=15, command=stop_video)
    btnStop.grid(row=0, column=1, padx=10)

def center_on_screen(win):

    win.update_idletasks()

    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


if __name__ == '__main__':
    """
    Main entry point of the script.
    """
    cap = None
    videoStreamingBox = None
    fpsBox = None
    
    network, class_names, class_colors = darknet.load_network("./nn/license-plate/license-plate.cfg", "./nn/license-plate/license-plate.data", "./nn/license-plate/license-plate.weights", batch_size=1)
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    frame_queue = Queue()
    frame_processed_queue = Queue()
    plate_ocr_queue = Queue()
    darknet_image_queue = Queue()
    detections_queue = Queue()
    frame_realtime_queue = Queue()
    #maxsize=1
    # Start threads and assign them to global variables
    detection_images_thread = Thread(target=detections_process, args=(darknet_image_queue, detections_queue))    
    drawing_thread = Thread(target=create_crops, args=(frame_queue, detections_queue))    
    drawing_ocr_thread = Thread(target=drawing_boxes_and_lincese_plate, args=(plate_ocr_queue,))    
    show_frame_processed_thread = Thread(target=show_frame_processed)

    splash_root = tk.Tk()
    splash_label = tk.Label(splash_root, text="Welcome to JustAnotherAlpr GUI", font=16)
    splash_label.pack(anchor="center", pady=40, padx=40)
    splash_root.overrideredirect(True)
    splash_root.attributes("-alpha", 0.8)

    center_on_screen(splash_root)

    splash_root.after(200, create_main_gui)

    splash_root.mainloop()

    # Flush and close standard outputs
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.close()
    sys.stderr.close()