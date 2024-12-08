
import os

import os
import shutil

import os
import shutil

def sort_mri_data():
    path = "../data/MRI/data"

    input_path = "../data/MRI/input"
    reference_path = "../data/MRI/reference"

    # Create the input and reference directories if they don't exist
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(reference_path, exist_ok=True)

    for case in os.listdir(path):
        if case.startswith("case"):
            # Extract the base case name (e.g., 'case1' from 'case1-ASHJDAS')
            base_case_name = case.split('-')[0]
            case_path = os.path.join(path, case)

            # Ensure the directories for this case exist in input and reference
            input_case_path = os.path.join(input_path, base_case_name)
            reference_case_path = os.path.join(reference_path, base_case_name)
            os.makedirs(input_case_path, exist_ok=True)
            os.makedirs(reference_case_path, exist_ok=True)

            for file in os.listdir(case_path):
                file_path = os.path.join(case_path, file)
                if "MR" in file:
                    # Move main nifti image to input/$case/
                    shutil.copy(file_path, os.path.join(input_case_path, file))
                elif "organ" in file and not "model" in file:
                    # Move corrected image to reference/$case/
                    shutil.copy(file_path, os.path.join(reference_case_path, file))

if __name__ == "__main__":
    sort_mri_data()
                