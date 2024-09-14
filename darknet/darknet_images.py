#!/usr/bin/env python3

import argparse
import os
import glob
import time
import cv2
import darknet

# Define a function named 'parser' to create and configure an argument parser
def parser():
    # Create an ArgumentParser object with a description
    parser = argparse.ArgumentParser(description="YOLO Object Detection")

    # Define command-line arguments and their descriptions
    # The default values are provided for each argument.

    # --input: Specifies the source of the images (single image, txt file with image paths, or folder)
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a txt with paths to them, or a folder. Image valid formats are jpg, jpeg, or png.")

    # --batch_size: Specifies the number of images to process at the same time
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")

    # --weights: Specifies the path to the YOLO weights file
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")

    # --dont_show: If provided, prevents displaying inference results in a window (useful for headless systems)
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")

    # --ext_output: If provided, displays bounding box coordinates of detected objects
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")

    # --save_labels: If provided, saves detections' bounding box coordinates in YOLO format for each image
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")

    # --config_file: Specifies the path to the YOLO configuration file
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")

    # --data_file: Specifies the path to the YOLO data file
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")

    # --thresh: Sets the confidence threshold for removing detections with lower confidence
    parser.add_argument("--thresh", type=float, default=.60,
                        help="remove detections with lower confidence")

    # --gpu: If provided, indicates the use of GPU for processing
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for processing")

    # --nms_thresh: Specifies the Non-Maximum Suppression threshold. If set, applies NMS.
    parser.add_argument("--nms_thresh", type=float, default=None,
                        help="Non-Maximum Suppression threshold. If set, applies NMS.")
    
    # --image_size: If provided, indicates the use the width and height of source imagen.
    parser.add_argument("--image_size", action="store_true", help="scale percent of image")

    # Parse the command-line arguments and return the result
    return parser.parse_args()

# This function defines the command-line arguments and their descriptions,
# and it returns the parsed arguments when called.

# Define a function named 'check_arguments_errors' that takes 'args' as input
def check_arguments_errors(args):
    # Check if the threshold value is within the valid range (0 < thresh < 1)
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"

    # Check if the specified YOLO configuration file exists
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))

    # Check if the specified YOLO weights file exists
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))

    # Check if the specified YOLO data file exists
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))

    # If '--input' is provided, check if the specified input image or file exists
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

# This function performs several checks on the command-line arguments and raises ValueError
# with appropriate error messages if any of the checks fail.

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

# Define a function named 'image_detection' that takes several arguments
def image_detection(image_or_path, network, class_names, class_colors, thresh, image_size):
    
    # Get the width and height of the Darknet network    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    # Check if 'image_or_path' is a path to an image file (string) or an image array
    if isinstance(image_or_path, str):
        # Load the image from the provided file path
        image = cv2.imread(image_or_path)

        # Check if the image loading was successful
        if image is None:
            raise ValueError(f"Unable to load image {image_or_path}")
    else:
        # Use the provided image array
        image = image_or_path

    if image_size:       
        width = int(image.shape[1] * 100 / 100)
        height = int(image.shape[0] * 100 / 100)

    # Create a Darknet IMAGE object with the specified width, height, and 3 channels
    darknet_image = darknet.make_image(width, height, 3)

    # Convert the input image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the dimensions of the Darknet network
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # Copy the resized image data into the Darknet IMAGE object
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # Perform object detection on the image using Darknet
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    
    # Free the memory used by the Darknet IMAGE object
    darknet.free_image(darknet_image)

    detections_adjusted = []
    
    if image_resized is not None and image_resized.size > 0:
        for label, confidence, bbox in detections:
            detections_adjusted.append((str(label), confidence, bbox))

        crops_resized = darknet.get_crops(detections_adjusted, image_resized)

        detection_count = 0

        for crop, bbox_adjusted in crops_resized:
            
            height, width = crop.shape[:2]

            if crop is not None and crop.size > 0 and  width > 0 and height > 0: 
                left, top, right, bottom = darknet.bbox2points(bbox_adjusted)                  
                image_resized, plate = darknet.read_lincese_plate_by_ocr(image_resized, crop, detections_adjusted[detection_count], left, top)
                detection_count += 1


    # Convert the image back to BGR color space (OpenCV format) and return it along with detections
    return cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), detections

# This function takes an image (either as a file path or an image array), a Darknet network,
# class names, class colors, and a detection threshold, and returns the image with bounding boxes
# and labels drawn around detected objects, as well as the list of detections.

# Function to convert bounding box coordinates to relative format
def convert2relative(image, bbox):
    """
    YOLO format uses normalized coordinates for annotation.

    Args:
        image: Input image (numpy array).
        bbox: Bounding box in absolute coordinates (x, y, width, height).

    Returns:
        Tuple representing bounding box coordinates in relative format (x_rel, y_rel, w_rel, h_rel).
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

# Function to save object detection annotations in YOLO format
def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates.

    Args:
        name: Name of the input image file.
        image: Input image (numpy array).
        detections: List of detected objects, each represented as (label, confidence, bbox).
        class_names: List of class names.

    Saves:
        Text file with YOLO-style annotations for object detection.
    """
    # Determine the output file name based on the input image name
    file_name = os.path.splitext(name)[0] + ".txt"

    # Open the output file for writing
    with open(file_name, "w") as f:
        # Iterate through detected objects
        for label, confidence, bbox in detections:
            # Convert bounding box coordinates to relative format
            x, y, w, h = convert2relative(image, bbox)

            # Find the index of the class label in class_names
            label = class_names.index(label)

            # Write annotation in YOLO format to the text file
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

# Function to perform object detection on images
def perform_detection(args, network, class_names, class_colors):
    # Load a list of image paths from the input directory
    images_paths = load_images(args.input)

    # Iterate over the image paths and perform object detection
    for image_path in images_paths:
        prev_time = time.time()

        # Perform image detection on the current image
        image, detections = image_detection(image_path, network, class_names, class_colors, args.thresh, args.image_size)

        # Print detections and calculate frames per second (FPS)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1 / (time.time() - prev_time))
        print("FPS: {}".format(fps))

        # Save annotations in YOLO format if requested
        if args.save_labels:
            save_annotations(image_path, image, detections, class_names)

        # Display the image with detections unless 'dont_show' is enabled
        if not args.dont_show:
            cv2.imshow('Inference', image)

            # Exit on 'q' key press
            if cv2.waitKey() & 0xFF == ord('q'):
                break

# Main function to handle object detection
def main():
    # Parse command line arguments using the 'parser' function
    args = parser()

    # Check for errors in the provided arguments
    check_arguments_errors(args)

    # If GPU is specified, set the GPU to use (GPU index 0 in this example)
    if args.gpu:
        darknet.set_gpu(0)

    # Load the Darknet network, class names, and class colors from configuration files
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    # Perform object detection on images using the loaded network and settings
    perform_detection(args, network, class_names, class_colors)

# Entry point of the script
if __name__ == "__main__":
    main()

