import csv

# Define the column names
columns = ['coronal', 'axial', 'sagittal']

# Define the file path
file_path = '../models/SIMPLE/planes.csv'

# Write the data to the CSV file
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)

print(f"CSV file '{file_path}' created successfully.")