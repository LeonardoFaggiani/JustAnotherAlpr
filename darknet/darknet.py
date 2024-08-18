#!/usr/bin/env python3

"""
Python 3 wrapper for identifying objects in images

Running the script requires opencv-python to be installed, via either of these two options:

- sudo apt-get install python3-opencv
- pip3 install opencv-python

NOTE:  As of December 2023 while using Ubuntu 22.04, the apt-get install one was
better since the pip3 one would result in a known segfault when cv2.imshow() was called:
https://github.com/opencv/opencv-python/issues/794

Directly viewing or returning bounding-boxed images requires scikit-image to be installed:

- pip3 install scikit-image

See the example code (such as "example_images.py") which imports this module.
"""

from ctypes import *
import os
import hashlib
import numpy as np
import re
from fast_plate_ocr import ONNXPlateRecognizer

# Global variable
colors_network = None
network = None

# Define a structure to represent a bounding box with (x, y, width, height)
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

# Define a structure to represent a detection result
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),             # Bounding box coordinates
                ("classes", c_int),        # Number of classes
                ("best_class_idx", c_int), # Index of the best class
                ("prob", POINTER(c_float)), # Array of class probabilities
                ("mask", POINTER(c_float)), # Mask for this detection
                ("objectness", c_float),   # Objectness score
                ("sort_class", c_int),     # Sorted class index
                ("uc", POINTER(c_float)),  # Uncertainty
                ("points", c_int),         # Number of points
                ("embeddings", POINTER(c_float)),  # Embeddings
                ("embedding_size", c_int), # Size of embeddings
                ("sim", c_float),          # Similarity score
                ("track_id", c_int)]       # Track ID

# Define a structure to represent a pair of detections
class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),           # Number of detections
                ("dets", POINTER(DETECTION))]  # Pointer to detection array

# Define a structure to represent an image
class IMAGE(Structure):
    _fields_ = [("w", c_int),            # Width of the image
                ("h", c_int),            # Height of the image
                ("c", c_int),            # Number of channels
                ("data", POINTER(c_float))]  # Pointer to image data

# Define a structure to represent metadata for classes
class METADATA(Structure):
    _fields_ = [("classes", c_int),        # Number of classes
                ("names", POINTER(c_char_p))]  # Pointer to class names

# Function to get the width of a network
def network_width(net):
    return lib.network_width(net)

# Function to get the height of a network
def network_height(net):
    return lib.network_height(net)

# Function to convert YOLO bounding box format to corner points in CV2 rectangle format
def bbox2points(bbox):
    """
    Convert bounding box from YOLO format to corner points in CV2 rectangle format.
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

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

    global network

    _height = network_height(network)
    _width = network_width(network)

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

def class_colors(names):
    """
    Create a dict with a deterministic, yet 'random', BGR color for each class name.
    This function uses a hash of the class name to generate consistent colors.
    """
    colors = {}
    for name in names:
        # Create a hash of the class name
        hash_object = hashlib.md5(name.encode())
        # Use the hash to generate a color
        hash_digest = hash_object.hexdigest()

        color = (int(hash_digest[0:2], 16), int(hash_digest[2:4], 16), int(hash_digest[4:6], 16))
        colors[name] = color

    return colors

def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    global colors_network
    global network

    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    colors_network = colors
    return network, class_names, colors

# Function to print detected objects and their confidence scores
def print_detections(detections, coordinates=False):
    """
    Print detected objects and their confidence scores.

    Args:
        detections: List of detected objects, each represented as (label, confidence, bbox).
        coordinates: If True, also print bounding box coordinates.

    Prints:
        Detected objects with or without bounding box coordinates.
    """
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

# Function to draw bounding boxes on an image
def draw_boxes(detection, image, colors):
    """
    Draw bounding boxes on an image.

    Args:
        detection: A detected object, represented as (label, confidence, bbox).
        image: Input image.
        colors: Dictionary of colors for each class label.

    Returns:
        Image with bounding boxes drawn.
    """
    
    import cv2
    
    label, confidence, bbox = detection
    
    left, top, right, bottom = bbox2points(bbox)

    cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
    cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        
    return image

# Function to decode detection results and format confidence scores
def decode_detection(detections):
    """
    Decode detection results and format confidence scores.

    Args:
        detections: List of detected objects, each represented as (label, confidence, bbox).

    Returns:
        List of decoded detection results with formatted confidence scores.
    """
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded

# Non-Maximum Suppression (NMS) function based on the method by Malisiewicz et al.
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(detections, overlap_thresh):
    """
    Apply non-maximum suppression (NMS) to a list of detections.

    Args:
        detections: List of detected objects, where each detection is represented as a tuple
                    (label, confidence, (x, y, w, h)).
        overlap_thresh: Threshold for considering overlap between bounding boxes.

    Returns:
        List of selected detections after NMS.
    """
    boxes = []

    # Convert detections to a list of bounding boxes
    for detection in detections:
        _, _, _, (x, y, w, h) = detection
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append(np.array([x1, y1, x2, y2]))

    boxes_array = np.array(boxes)

    # Initialize the list of picked indexes
    pick = []

    # Extract coordinates of the bounding boxes
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]

    # Compute the area of the bounding boxes and sort by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add it to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have high overlap with the current box
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return [detections[i] for i in pick]

# Function to remove all classes with 0% confidence within the detection
def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection.

    Args:
        detections: List of detections, where each detection is an object with confidence scores for classes.
        class_names: List of class names.
        num: Number of detections.

    Returns:
        List of valid predictions with class name, confidence, and bounding box.
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions

# Faster version of remove_negatives (very useful when using yolo9000)
def remove_negatives_faster(detections, class_names, num):
    """
    Faster version of remove_negatives (very useful when using yolo9000).

    Args:
        detections: List of detections, where each detection is an object with confidence scores for classes.
        class_names: List of class names.
        num: Number of detections.

    Returns:
        List of valid predictions with class name, confidence, and bounding box.
    """
    predictions = []
    for j in range(num):
        if detections[j].best_class_idx == -1:
            continue
        name = class_names[detections[j].best_class_idx]
        bbox = detections[j].bbox
        bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
        predictions.append((name, detections[j].prob[detections[j].best_class_idx], bbox))
    return predictions

# Function to perform object detection on an input image using a Darknet network
def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
    Returns a list with the highest confidence class and their bounding box.

    Args:
        network: Darknet network.
        class_names: List of class names.
        image: Input image.
        thresh: Detection confidence threshold.
        hier_thresh: Hierarchical threshold.
        nms: Non-Maximum Suppression threshold.

    Returns:
        List of detections with class name, confidence, and bounding box.
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])

# Function to read license plate
def read_lincese_plate_by_ocr(image, crop, detection_adjusted, left, top):
    """
    Return the frame with the license-plate has red

    Parameters:
    image: the current frame to write boxes and license-plate
    crop: crop of lincese-plate.
    detection_adjusted: the detection of lincese-plate (label, condifence, bbox)
    left: position that start the letters of license-plate
    top: position that start the letters of license-plate

    This function return the final frame with the box of license-plate detected and ORC result unless
    the average of OCR result is under 65% of condifence 
    """
    import cv2
    
    global colors_network

    result, confidences = plate_recognizer.run(crop, True)

    print(result, confidences)

    average = np.mean(confidences)

    if result is not None and average > 0.65:
        # Draw bounding boxes on the frame
        image = draw_boxes(detection_adjusted, image, colors_network)
        cv2.putText(image, validate_and_correct_format_license_plate(result[0]), (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

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

        left, top, right, bottom = bbox2points(bbox_adjusted)  

        height, width = image.shape[:2]

        width = int((width  * scale_percent / 100))
        height = int((height  * scale_percent / 100))   

        crop = image[top-5:bottom+5, left-5:right+5]

        if crop is not None and crop.size > 0:
            image_resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
            image_grey = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            crops_resized.append((image_grey, bbox_adjusted))

    return crops_resized

# Function to valida Argentinian format license plate
def validate_and_correct_format_license_plate(plate):
    
    # Regex to validate Argentian plate format
    pattern = r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$|^[A-Z]{3}[0-9]{3}_$'
    
    match = re.match(pattern, plate)
    
    if match:
        # If match with the second format then remove '_'
        if plate.endswith('_'):
            plate = plate.replace('_', '')
        return plate
    else:
        return 'Invalid'
    
# Platform-specific library path and initialization
if os.name == "posix":
    libpath = "/usr/lib/libdarknet.so"
elif os.name == "nt":
    libpath = "path/to/your/darknet.dll"
else:
    print("Unsupported OS")
    exit
lib = CDLL(libpath, RTLD_GLOBAL)

# Argument types and return types for Darknet functions
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int


# Define and comment function to copy image data from bytes to a Darknet IMAGE object
copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

# Define and comment function to predict using a Darknet network
predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

# Define and comment function to set the GPU device for Darknet
set_gpu = lib.cuda_set_device

# Define and comment function to initialize Darknet for CPU processing
init_cpu = lib.init_cpu

# Define and comment function to create a Darknet IMAGE object
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

# Define and comment function to get network boxes for detections
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

# Define and comment function to create network boxes
make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

# Define and comment function to free detections
free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

# Define and comment function to free pointers
free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

# Define and comment function to predict using a Darknet network
network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

# Define and comment function to reset RNN (Recurrent Neural Network)
reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

# Define and comment function to load a Darknet network
load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

# Define and comment function to load a custom Darknet network
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

# Define and comment function to free a network pointer
free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

# Define and comment function to perform Non-Maximum Suppression (NMS) on object detections
do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# Define and comment function to perform NMS with sorted results
do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# Define and comment function to free a Darknet image
free_image = lib.free_image
free_image.argtypes = [IMAGE]

# Define and comment function to letterbox a Darknet image
letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

# Define and comment function to load metadata for Darknet
load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

# Define and comment function to load a color image for Darknet
load_image = lib.load_image
load_image.argtypes = [c_char_p, c_int, c_int, c_int]
load_image.restype = IMAGE

# Define and comment function to convert RGB image to BGR
rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

# Define and comment function to predict using an image and a Darknet network
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

# Define and comment function to predict using a letterboxed image and a Darknet network
predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

# Define and comment function to predict using a batch of images and a Darknet network
network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

# Define plates OCR
plate_recognizer = ONNXPlateRecognizer('argentinian-plates-cnn-model')
