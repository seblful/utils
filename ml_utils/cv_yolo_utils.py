import os
import cv2
import random


def create_images_labels_dict(images_path: str,
                              shuffle: bool = False) -> dict:
    '''
    Takes images folder path and returns dict where each image
    corresponds each label and shuffles if True

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


def read_bbox_file(label_path):
    '''
    Reads file with bboxes and returns list with bboxes'''
    std_bboxes = []

    with open(label_path, 'r') as lab:
        for line in lab:
            bbox = [float(i) for i in line.split()][1:]
            std_bboxes.append(bbox)
    return std_bboxes


def get_min_and_max_coords(bbox,
                           image_width,
                           image_height):
    """Takes yolo format bboxes and returns start point and end point"""

    # Extracting each point from labels
    center_x = bbox[0] * image_width
    center_y = bbox[1] * image_height
    width = bbox[2] * image_width
    height = bbox[3] * image_height

    # Converting points to opencv format
    xmin = int(center_x - (width / 2))
    ymin = int(center_y - (height / 2))
    xmax = int(center_x + (width / 2))
    ymax = int(center_y + (height / 2))

    return (xmin, ymin), (xmax, ymax)


def draw_rectangle(image,
                   start_point,
                   end_point,
                   bbox_color=(255, 255, 255),
                   thickness=2):
    """Draws rectangles"""

    # Drawing rectangle on an image
    image = cv2.rectangle(image, start_point, end_point,
                          bbox_color, thickness)

    return image


def visualize_bbox(data_path='raw-data', show_only_mul_ob=False):
    '''
    Takes folder with images and labels and visualizes it
    '''

    # Creating paths with images and labels
    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    # Iterating through each images and labels
    for image_name, label_name in create_images_labels_dict(images_path=images_path,
                                                            shuffle=True).items():
        print(image_name, label_name)
        # Creating paths with labels
        label_full_path = os.path.join(labels_path, label_name)

        # Extracting bboxes from text files
        std_bboxes = read_bbox_file(label_full_path)

        # Show only images with multiple objects
        if show_only_mul_ob and len(std_bboxes) in [0, 1]:
            continue

        # Creating paths with images and read images
        image_full_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_full_path)
        image_height, image_width = image.shape[:2]

        # Draw bboxes on image
        for bbox in std_bboxes:
            start_point, end_point = get_min_and_max_coords(
                bbox, image_width, image_height)
            image = draw_rectangle(image, start_point, end_point)

        image = cv2.resize(image, (1280, 720))
        cv2.imshow('Image with Polygons', image)
        cv2.waitKey(0)


def check_image_label_pairs(images_folder, labels_folder):
    images = [img for img in os.listdir(
        images_folder) if img.endswith(('.jpg', '.png'))]
    labels = [label for label in os.listdir(
        labels_folder) if label.endswith('.txt')]

    for image in images:
        image_name = os.path.splitext(image)[0]
        label_name = image_name + '.txt'
        if label_name not in labels:
            print(f"Label missing for image: {image}")


def modify_label_files(folder_path):
    # List all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Iterate through the .txt files
    for file in txt_files:
        # Read the content of the file
        full_file_path = os.path.join(folder_path, file)
        with open(full_file_path, 'r') as file_content:
            lines = file_content.readlines()

        # Modify the content of the file
        for i, line in enumerate(lines):
            if line.split()[0] != '0':
                lines[i] = '0 ' + line.split(None, 1)[1]

        # Write the modified content back to the file
        with open(full_file_path, 'w') as file_content:
            file_content.writelines(lines)


def check_lines_start_with_0(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            full_file_path = os.path.join(folder_path, file)
            with open(full_file_path, 'r') as f:
                lines = f.readlines()
                all_lines_start_with_0 = all(
                    line.startswith('0') for line in lines)
                if not all_lines_start_with_0:
                    print(f"Not all lines in {file} start with a 0.")


def main():
    # visualize_bbox(data_path='data', show_only_mul_ob=False)
    # check_image_label_pairs('data/images', 'data/labels')
    # modify_label_files('data/labels')
    check_lines_start_with_0('data/labels')
    pass


if __name__ == '__main__':
    main()
