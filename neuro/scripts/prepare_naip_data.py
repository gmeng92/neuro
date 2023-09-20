# prepare the naip data 
# Given the data path and the number of labels,
# generate a dataframe with the path to the images and the path to the labels
# and save the dataframe to csv files
#
# Usage:


#

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


def find_sample(path, seed_val = 123, train_size = 0.7, valid_size = 0.1):
    """
    Args:
        path (str): path to mri images
            the path to the mri images is in the following format:
                subject_UID/resampled_t1.nii.gz
                subject_UID/segmentation_labels.nii.gz
            labels = [  0, # outside of the brain
                        1, # gray matter
                        2, # white matter
                        3  # cerebrospinal fluid
                        ]
        seed_val (int): random seed for reproducibility
        train_size (float): percentage of the training data
        valid_size (float): percentage of the validation data
        Remark: train_size + valid_size < 1, and the rest is the inference data
    """
    labels_data = {"images": [], "labels": []}
    t = 0
    for person in os.listdir(path):
            if person.startswith("."):
                continue
            # mri_images/subject_UID
            person_folder = os.path.join(path, person)
            for train in os.listdir(person_folder):
                if train.startswith("."):
                    continue
                if train == "resampled_t1.nii.gz":
                    labels_data["images"].append(os.path.join(person_folder, "resampled_t1.nii.gz"))
                if train == "segmentation_labels.nii.gz":
                    labels_data["labels"].append(
                        os.path.join(person_folder, "segmentation_labels.nii.gz")
                    )
                    t += 1
    print(f"Total of {t} subjects!")

    dataframe = pd.DataFrame(labels_data)
    dataframe.to_csv("./data/dataset.csv", index=False)
    dataframe = dataframe.sample(frac=1, random_state=seed_val)
    # set the percentage of the training, validation and inference data
    if train_size+valid_size >= 1:
        raise ValueError("train_size + valid_size must be less than 1")
    else:
        num_train = int(len(dataframe) * train_size)
        num_valid = int(len(dataframe) * valid_size)
        dataframe.iloc[:num_train, :].to_csv("./data/dataset_train.csv", index=False)
        dataframe.iloc[num_train:num_train+num_valid, :].to_csv("./data/dataset_valid.csv", index=False)
        dataframe.iloc[num_train+num_valid:, :].to_csv("./data/dataset_infer.csv", index=False)
        print(f'Saving the dataset split to ./data/dataset_train.csv, ./data/dataset_valid.csv, ./data/dataset_infer.csv...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    parser.add_argument("seed_val", type=int, default=123, help="random seed for reproducibility")
    parser.add_argument("train_size", type=float, default=0.7, help="percentage of the training data")
    parser.add_argument("valid_size", type=float, default=0.1, help="percentage of the validation data")
    params = parser.parse_args()

    find_sample(params.datapath, params.n_labels, params.seed_val, params.train_size, params.valid_size)

