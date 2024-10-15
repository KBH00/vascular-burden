import pandas as pd
import os

def find_subject_directories(csv_path, root_dir):
    df = pd.read_csv(csv_path, sep=',', quotechar='"') 
    print("Columns in the DataFrame:", df.columns)

    filtered_df = df[df['PXPERIPH'] == 2]

    subject_dirs = []
    for subject_id in filtered_df['subject_id']:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if subject_id in dirnames:
                subject_dirs.append(os.path.join(dirpath, subject_id))

    return subject_dirs

def find_anomal():
    csv_path = 'D:/Data/vsclr_csv/All Subjects_Peripheral Vascular.csv'  
    root_dir = 'D:/VascularData/data/nii'  
    directories = find_subject_directories(csv_path, root_dir)
    print(directories)

#find_anomal()