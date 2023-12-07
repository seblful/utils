from typing import List, Dict

import os
import random
import numpy as np
import cv2


def create_images_labels_dict(images_path: str,
                              shuffle: bool = False) -> Dict[str, str]:
    '''
    Takes images folder path and returns dict where each image
    corresponds each label and shuffles if True.

    Folder with images should contain folder with labels 
    with name 'labels'.
    '''

    # Iterate throgh images_path and create a dictionary to store the images and labels names
    images_labels_dict = {}
    for image_name in os.listdir(images_path):
        label_name = os.path.splitext(image_name)[0] + '.txt'

        images_labels_dict[image_name] = label_name

    # Shuffle the dict
    if shuffle:
        keys = list(images_labels_dict.keys())
        random.shuffle(keys)
        images_labels_dict = {key: images_labels_dict[key] for key in keys}

    return images_labels_dict


def read_yolo_bbox_file(label_path: str) -> List[List[float]]:
    '''
    Reads file with yolo bboxes and returns list with lists with bboxes.
    '''
    # Create list for storing bboxes
    yolo_bboxes = []

    # Read label file, create list with bbox and add to yolo_bboxes list
    with open(label_path, 'r') as file:
        for line in file:
            yolo_bbox = [float(i) for i in line.split()][1:]
            yolo_bboxes.append(yolo_bbox)

    return yolo_bboxes


def get_min_and_max_coords(bbox: List[float],
                           image_width: int,
                           image_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Takes list with yolo bbox, transform it to OpenCV format 
    (start point and end points) and returns tuple with 
    (xmin, ymin), (xmax, ymax) values.
    """

    # Extracting each point from labels
    center_x = bbox[0] * image_width
    center_y = bbox[1] * image_height
    bbox_width = bbox[2] * image_width
    bbox_height = bbox[3] * image_height

    # Converting points to OpenCV format
    xmin = int(center_x - (bbox_width / 2))
    ymin = int(center_y - (bbox_height / 2))
    xmax = int(center_x + (bbox_width / 2))
    ymax = int(center_y + (bbox_height / 2))

    return (xmin, ymin), (xmax, ymax)


def draw_rectangle(image: np.ndarray,
                   start_point: tuple[int, int],
                   end_point: tuple[int, int],
                   bbox_color: tuple[int] = (255, 0, 0),
                   thickness: int = 2) -> np.ndarray:
    '''
    Takes image array, start point and end OpenCV points
    and draws rectangle on image with specific color and 
    thickness.
    '''

    # Drawing rectangle on an image
    image = cv2.rectangle(image, start_point, end_point,
                          bbox_color, thickness)

    return image


def visualize_bbox(data_path: str,
                   show_only_mul_ob: bool = False) -> np.ndarray:
    '''
    Takes data folder path that contains images and labels folder,
    iterating through each image and label, reads it, processes,
    shows visualization and returns processed image.
    '''
    # Creating paths with images and labels folders
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    # Iterating through each image_name and label_name
    for image_name, label_name in create_images_labels_dict(images_path=images_path,
                                                            shuffle=True).items():
        # Creating full paths with image and label
        image_full_path = os.path.join(images_path, image_name)
        label_full_path = os.path.join(labels_path, label_name)

        # Extracting yolo bboxes from text files
        std_bboxes = read_yolo_bbox_file(label_full_path)

        # Show only images with multiple objects
        if show_only_mul_ob and len(std_bboxes) in [0, 1]:
            continue

        # Read image and extrach image_height and image_width
        image = cv2.imread(image_full_path)
        image_height, image_width = image.shape[:2]

        # Draw bboxes on image
        for bbox in std_bboxes:
            start_point, end_point = get_min_and_max_coords(
                bbox, image_width, image_height)
            image = draw_rectangle(image, start_point, end_point)

        # Resize image keeping aspect ratio
        # image = cv2.resize(image, (1280, 720))
        image = resize_image_with_padding(image, (640, 640))
        cv2.imshow('Bbox Visualizer', image)

        # Break if ['q', 'Q'] is pressed
        if cv2.waitKey(0) in [ord('q'), ord('Q')]:
            break

    return image


def resize_image_with_padding(image: np.ndarray,
                              shape_out: tuple[int, int],
                              padding: bool = True,
                              tiny_float: float = 1e-5) -> np.ndarray:
    """
    Resizes an image to the specified size, adding padding 
    to preserve the aspect ratio and returns the image.
    """
    # Calculate resize ratio
    if image.ndim == 3 and len(shape_out) == 2:
        shape_out = [*shape_out, 3]
    hw_out, hw_image = [np.array(x[:2]) for x in (shape_out, image.shape)]
    resize_ratio = np.min(hw_out / hw_image)
    hw_wk = (hw_image * resize_ratio + tiny_float).astype(int)

    # Resize the image
    resized_image = cv2.resize(
        image, tuple(hw_wk[::-1]), interpolation=cv2.INTER_NEAREST
    )
    if not padding or np.all(hw_out == hw_wk):
        return resized_image

    # Create a black image with the target size
    padded_image = np.zeros(shape_out, dtype=np.uint8)

    # Calculate the number of rows/columns to add as padding
    dh, dw = (hw_out - hw_wk) // 2
    # Add the resized image to the padded image, with padding on the left and right sides
    padded_image[dh: hw_wk[0] + dh, dw: hw_wk[1] + dw] = resized_image

    return padded_image


def check_image_label_pairs(data_path: str) -> int:
    '''
    Iterates through images and labels and checks 
    if label pair is missing for image.
    '''
    # Creating paths with images and labels folders
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    # Listdir images names
    images_names = [img for img in os.listdir(
        images_path) if img.endswith(('.jpg', '.png'))]
    labels_names = [label for label in os.listdir(
        labels_path) if label.endswith('.txt')]

    # Create counter for missing labels
    missing_labels_counter = 0

    # Iterating through images_names and check if it has label pair
    for image_name in images_names:
        image_name_split = os.path.splitext(image_name)[0]
        label_name = image_name_split + '.txt'
        if label_name not in labels_names:
            print(f"Label missing for image: {image_name}")

    print(f"{missing_labels_counter} images have not label pair.")

    return missing_labels_counter


def modify_label_files(folder_path: str,
                       allow_class_labels: List[int] = [0]) -> None:
    '''
    Iterates through each yolo label bbox file, checks if 
    class in allow_class_labels and change it to 0 if not.
    '''
    # List all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Iterate through the .txt files
    for file_name in txt_files:
        # Create full file path
        full_file_path = os.path.join(folder_path, file_name)

        # Read the content of the file and change class if it is not starts with allow_class_labels
        with open(full_file_path, 'r') as file:
            lines = file.readlines()
            # Modify the content of the file
            for i, line in enumerate(lines):
                if int(line.split()[0]) not in allow_class_labels:
                    print(
                        f"Not all lines in {file_name} start with allow class labels.")
                    lines[i] = '0 ' + line.split(None, 1)[1]

        # Write the modified content back to the file
        with open(full_file_path, 'w') as file:
            file.writelines(lines)


def main():
    visualize_bbox('data')


if __name__ == '__main__':
    main()
