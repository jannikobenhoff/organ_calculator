import os
import shutil

def split_nnunet_data(combined_data_dir="../data/totalsegmentator_combined", output_dir="../data/nnUNet_raw/Dataset101_Totalsegmentator", training_split=0.8):
    images = os.listdir(combined_data_dir + "/imagesTr")
    labels = os.listdir(combined_data_dir + "/labelsTr")

    print("Number of images:", len(images))
    print("Number of labels:", len(labels))

    # Split data
    num_training = int(training_split * len(images))

    training_images = images[:num_training]
    training_labels = labels[:num_training]

    test_images = images[num_training:]
    test_labels = labels[num_training:]

    # Move files to outut directory
    os.makedirs(output_dir + "/imagesTr", exist_ok=True)
    os.makedirs(output_dir + "/imagesTs", exist_ok=True)
    os.makedirs(output_dir + "/labelsTr", exist_ok=True)
    os.makedirs(output_dir + "/labelsTs", exist_ok=True)

    for image in training_images:
        shutil.copy(combined_data_dir + "/imagesTr/" + image, output_dir + "/imagesTr/" + image)
    
    for label in training_labels:
        shutil.copy(combined_data_dir + "/labelsTr/" + label, output_dir + "/labelsTr/" + label)

    for image in test_images:
        shutil.copy(combined_data_dir + "/imagesTr/" + image, output_dir + "/imagesTs/" + image)

    for label in test_labels:
        shutil.copy(combined_data_dir + "/labelsTr/" + label, output_dir + "/labelsTs/" + label)

    print("Data split successfully!")

def check_nnunet_data():
    training_images = os.listdir("../data/nnUNet_raw/Dataset101_Totalsegmentator/imagesTr")
    training_labels = os.listdir("../data/nnUNet_raw/Dataset101_Totalsegmentator/labelsTr")
    test_images = os.listdir("../data/nnUNet_raw/Dataset101_Totalsegmentator/imagesTs")
    test_labels = os.listdir("../data/nnUNet_raw/Dataset101_Totalsegmentator/labelsTs")

    print("Number of training images:", len(training_images))
    print("Number of training labels:", len(training_labels))
    print("Number of test images:", len(test_images))
    print("Number of test labels:", len(test_labels))

def rename_split():
    """
    images: {name}_001_0000.nii.gz
    labels: {name}_001.nii.gz
    """
    image_dir = "../data/nnUNet_raw/Dataset101_Totalsegmentator/imagesTr"
    label_dir = "../data/nnUNet_raw/Dataset101_Totalsegmentator/labelsTr"

    index = 0
    for i, image in enumerate(os.listdir(image_dir)):
        os.rename(image_dir + "/" + image, image_dir + f"/ts_{i:03d}_0000.nii.gz")
    
    for i, label in enumerate(os.listdir(label_dir)):
        os.rename(label_dir + "/" + label, label_dir + f"/ts_{i:03d}.nii.gz")
        index += 1

    image_dir = "../data/nnUNet_raw/Dataset101_Totalsegmentator/imagesTs"
    label_dir = "../data/nnUNet_raw/Dataset101_Totalsegmentator/labelsTs"

    for i, image in enumerate(os.listdir(image_dir)):
        i += index
        os.rename(image_dir + "/" + image, image_dir + f"/ts_{i:03d}_0000.nii.gz")
    
    for i, label in enumerate(os.listdir(label_dir)):
        i += index
        os.rename(label_dir + "/" + label, label_dir + f"/ts_{i:03d}.nii.gz")
    

if __name__ == "__main__":
    rename_split()
    # Number of training images: 982
    # Number of training labels: 982
    # Number of test images: 246
    # Number of test labels: 246