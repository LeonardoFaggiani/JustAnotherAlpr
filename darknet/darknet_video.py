#!/usr/bin/env python3

from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import signal
import sys
import numpy as np

# Global flag to control the execution of threads
is_running = True

# Global variables for threads and speed control
capture_thread = None
inference_thread = None
drawing_thread = None
plate_ocr_queue = None
frame_delay = 2  # Delay in milliseconds (default is 30 for normal speed)

def parse_args():
    # Create an ArgumentParser object with a description for YOLO Object Detection.
    parser = argparse.ArgumentParser(description="YOLO Object Detection")

    # Add argument for input video source. Default is '0', which typically refers to the default webcam.
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")

    # Add argument for the output file name. Default is an empty string, indicating no file will be saved.
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")

    # Add argument for the path to the YOLO weights file. Default is set to 'yolov4.weights'.
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")

    # Add boolean argument to not display the window during inference. Useful for headless systems.
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")

    # Add boolean argument to display bounding box coordinates on detected objects.
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")

    # Add argument for the path to the YOLO configuration file. Default is set to 'yolov4.cfg'.
    parser.add_argument("--config_file", default="yolov4.cfg",
                        help="path to config file")

    # Add argument for the path to the data file. Default is set to 'coco.data'.
    parser.add_argument("--data_file", default="coco.data",
                        help="path to data file")

    # Add argument for the threshold to filter detections. Default is set to 0.75.
    parser.add_argument("--thresh", type=float, default=.75,
                        help="remove detections with confidence below this value")

    # Add argument for specifying the GPU index to use. Default is '0'.
    parser.add_argument("--gpu_index", type=int, default=0,
                        help="GPU index to use for processing")

    # Parse the arguments received from the command line and return them.
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    # Assert that the threshold value is between 0 and 1 (exclusive). This is to ensure that the threshold
    # for detecting objects is set to a sensible value.
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"

    # Check if the YOLO configuration file exists at the specified path. Raise an error if it doesn't.
    # This file is necessary for setting up the YOLO network architecture.
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))

    # Check if the YOLO weights file exists at the specified path. Raise an error if it doesn't.
    # This file contains the trained model weights necessary for object detection.
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))

    # Check if the data file exists at the specified path. Raise an error if it doesn't.
    # This file typically contains information such as class labels.
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))

    # If the input argument is a string (path to a video file), check if the file exists.
    # This is to ensure that the video file for detection is available.
    # The str2int function attempts to convert the input argument to an integer (for webcam input).
    # If this conversion fails, the input is treated as a file path, which should exist.
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    # Define the video codec using the FourCC code.
    # 'MJPG' is used here which stands for Motion-JPEG codec.
    # This codec is widely used and generally provides a good balance between quality and file size.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # Retrieve the frames per second (fps) from the input video.
    # This is important to maintain the same temporal characteristics in the output video as the input.
    fps = int(input_video.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object that will be used to write the output video.
    # The parameters include the output file name, the codec (fourcc), the fps, and the frame size.
    # The frame size is specified as a tuple (width, height), and it should match the frames being written.
    video = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Return the VideoWriter object so it can be used to write frames to the output video file.
    return video


def convert2relative(bbox):
    """
    Converts bounding box coordinates from absolute to relative values.
    In YOLO, relative coordinates are used where the dimensions of the image are normalized to a range of 0 to 1.

    Parameters:
    bbox (tuple): A tuple containing the bounding box's absolute coordinates in the format (x, y, w, h).
                  Here, x and y represent the center of the box, while w and h are its width and height.

    Returns:
    tuple: A tuple containing the bounding box's relative coordinates.
    """

    # Unpack the bounding box coordinates
    x, y, w, h = bbox

    # The network width and height are used to normalize the coordinates.
    # These should be the dimensions of the image as used by the network, usually the input layer dimensions.
    _height = darknet_height
    _width = darknet_width

    # Normalize the coordinates by dividing by the width and height of the image.
    # This converts the coordinates from a pixel value to a relative value based on the size of the image.
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image, bbox):
    """
    Converts relative bounding box coordinates back to original (absolute) coordinates.

    Parameters:
    image (array): The original image array on which detection was performed.
    bbox (tuple): A tuple containing the bounding box's relative coordinates (x, y, w, h)
                  where x and y are the center of the box, and w and h are width and height.

    Returns:
    tuple: A tuple containing the bounding box's original (absolute) coordinates.
    """

    # First, convert relative coordinates (normalized) back to absolute coordinates.
    # This is done by multiplying the relative coordinates by the actual image dimensions.
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape  # Get the height and width of the original image.

    # Calculate the original x and y coordinates (top-left corner of the bounding box).
    orig_x = int(x * image_w)
    orig_y = int(y * image_h)

    # Calculate the original width and height of the bounding box.
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    # The resulting bounding box in original image coordinates.
    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    """
    Converts relative bounding box coordinates to absolute coordinates suitable for cropping.

    Parameters:
    image (array): The original image array on which detection was performed.
    bbox (tuple): A tuple containing the bounding box's relative coordinates (x, y, w, h)
                  where x and y are the center of the box, and w and h are width and height.

    Returns:
    tuple: A tuple containing the coordinates for cropping (left, top, right, bottom).
    """

    # Convert relative coordinates to absolute coordinates using image dimensions.
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape

    # Calculate the absolute coordinates for the left, right, top, and bottom edges
    # of the bounding box. These are used for cropping the image.
    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    # Ensure that the coordinates do not exceed the image boundaries.
    # This is important as attempting to crop outside the image dimensions can cause errors.
    if orig_left < 0: orig_left = 0
    if orig_right > image_w - 1: orig_right = image_w - 1
    if orig_top < 0: orig_top = 0
    if orig_bottom > image_h - 1: orig_bottom = image_h - 1

    # The coordinates for cropping the image.
    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def signal_handler(sig, frame):
    """
    Handles a specific signal and stops the main loop.
    """
    global is_running, capture_thread, inference_thread, drawing_thread, drawing_ocr_thread
    is_running = False

    # Join threads
    if capture_thread is not None:
        capture_thread.join()
    if inference_thread is not None:
        inference_thread.join()
    if drawing_thread is not None:
        drawing_thread.join()
    if drawing_ocr_thread is not None:
        drawing_ocr_thread.join()


def video_capture(frame_queue, darknet_image_queue):
    global is_running, frame_delay

    while is_running and cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video source
        if not ret:  # If no frame is captured, break the loop
            break

        # Handle key presses for speed control
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            is_running = False

        # Convert the frame from BGR (OpenCV default) to RGB
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

    # Release the video source when the loop ends
    cap.release()


def inference(darknet_image_queue, detections_queue):
    """
    Processes images using the Darknet YOLO framework for object detection.

    Parameters:
    darknet_image_queue: A queue containing images to be processed.
    detections_queue: A queue to put the detections for later use.

    This function continuously retrieves images from the darknet_image_queue,
    performs object detection on them, and puts the detections into his respective queues.
    This allows for parallel processing and data handling
    in other threads.
    """
    global is_running
    while is_running and cap.isOpened():
        darknet_image = darknet_image_queue.get()  # Retrieve an image from the queue

        # Perform object detection on the image
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)

        detections_queue.put(detections)  # Put the detections in the detections queue

        # Print detections if extended output is enabled
        #darknet.print_detections(detections, args.ext_output)

        darknet.free_image(darknet_image)  # Free the memory of the Darknet image

    # Release the video source when the loop ends
    cap.release()


def create_crops(frame_queue, detections_queue):
    """
    Create crops for each detection of license-plate.

    Parameters:
    frame_queue: Queue from which to retrieve frames.
    detections_queue: Queue from which to retrieve detections for each frame.

    This function continuously retrieves frames and their corresponding detections,
    then create the crops for every detection and saves it into a queue for OCR.
    """
    global is_running
    random.seed(3)  # Ensure consistent colors for bounding boxes across runs



    while is_running and cap.isOpened():
        frame = frame_queue.get()  # Retrieve a frame from the queue
        detections = detections_queue.get()  # Retrieve detections for the frame

        detections_adjusted = []

        if frame is not None and frame.size > 0:
            # Adjust each detection to the original frame size and add to list
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            
            crops_resized = get_crops(detections_adjusted, frame)

            ocr_items = (crops_resized, frame, detections_adjusted)

            plate_ocr_queue.put(ocr_items)
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def drawing_boxes_and_lincese_plate(plate_ocr_queue):
    """
    Draws bounding boxes on frames and displays them.

    Parameters:
    plate_ocr_queue: Queue from which to retrieve detections for each crops and their corresponding bbox.

    This function continuously retrieves the queue of crops and their corresponding bbox,
    if an output filename is provided, it also writes the frames to a video file.
    """

    import cv2
    global is_running

    # Initialize video writer if an output filename is specified
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))

    while is_running and cap.isOpened():

        cropsAndBbox, frame, detections_adjusted = plate_ocr_queue.get()

        for crop, bbox_adjusted in cropsAndBbox:

            height, width = crop.shape[:2]

            if crop is not None and crop.size > 0 and  width > 0 and height > 0: 
                left, top, right, bottom = darknet.bbox2points(bbox_adjusted)  
                frame = read_lincese_plate_by_ocr(frame, crop, detections_adjusted, left, top)
        
        if frame is not None:    
            if not args.dont_show:
                cv2.imshow('Inference', frame)  # Display the frame

                # Check if the 'q' key is pressed to stop the process
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    is_running = False
                    break
        
        # Write the frame to the output video file if specified
        if args.out_filename is not None:
            video.write(frame)

    video.release()
            
# Function to read license plate
def read_lincese_plate_by_ocr(image, crop, detections_adjusted, left, top):

    """
    Return the frame with the license-plate has red

    Parameters:
    image: the current frame to write boxes and license-plate
    crop: crop of lincese-plate.
    detections_adjusted: the detection of lincese-plate (label, condifence, bbox)
    left: position that start the letters of license-plate
    top: position that start the letters of license-plate

    This function return the final frame with the box of license-plate detected and ORC result unless
    the average of OCR result is under 65% of condifence 
    """
    import cv2

    result, confidences = darknet.plate_recognizer.run(crop, True)

    average = np.mean(confidences)

    if result is not None and average > 0.65:
        # Draw bounding boxes on the frame    
        image = darknet.draw_boxes(detections_adjusted, image, class_colors)                   
        cv2.putText(image, result[0], (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return image

# Function to create crops
def get_crops(detections, image, scale_percent = 3):

    """
    Return the crops resized and in grey format for OCR with the corresponding bbox

    Parameters:
    detections: Queue from which to retrieve detections for each crops and their corresponding bbox.
    image: frames.
    scale_percent: Scale in percent to resize the crops.

    This function return the crops resized and in grey format.
    """
    import cv2

    crops_resized = []

    for label, confidence, bbox_adjusted in detections:

        left, top, right, bottom = darknet.bbox2points(bbox_adjusted)  

        height, width = image.shape[:2]

        width = int((width  * scale_percent / 100))
        height = int((height  * scale_percent / 100))   

        crop = image[top-5:bottom+5, left-5:right+5]

        if crop is not None and crop.size > 0:
            image_resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
            image_grey = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            crops_resized.append((image_grey, bbox_adjusted))

    return crops_resized


if __name__ == '__main__':
    """
    Main entry point of the script.
    """

    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command-line arguments
    args = parse_args()

    # Set GPU, perform checks, load network as before
    darknet.set_gpu(args.gpu_index)
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file, args.data_file, args.weights, batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_queue = Queue()
    plate_ocr_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)

    # Start threads and assign them to global variables
    capture_thread = Thread(target=video_capture, args=(frame_queue, darknet_image_queue))
    inference_thread = Thread(target=inference, args=(darknet_image_queue, detections_queue))
    drawing_thread = Thread(target=create_crops, args=(frame_queue, detections_queue))
    drawing_ocr_thread = Thread(target=drawing_boxes_and_lincese_plate, args=(plate_ocr_queue,))

    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()
    drawing_ocr_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    inference_thread.join()
    drawing_thread.join()
    drawing_ocr_thread.join()

    # Flush and close standard outputs
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.close()
    sys.stderr.close()

