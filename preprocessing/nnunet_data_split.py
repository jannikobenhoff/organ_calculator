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

if __name__ == "__main__":
    # split_nnunet_data()
    check_nnunet_data()