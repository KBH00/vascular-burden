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

def find_subject_id(csv_path, root_dir):
    df = pd.read_csv(csv_path, sep=',', quotechar='"') 
    print("Columns in the DataFrame:", df.columns)

    filtered_df = df[df['PXPERIPH'] == 2]
    print(filtered_df)
    subject_dirs = []
    for subject_id in filtered_df['subject_id']:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if subject_id in dirnames:
                subject_dirs.append(subject_id)

    return subject_dirs

def find_anomal():
    csv_path = 'D:/Download/abnormal.csv'  
    root_dir = 'D:/Download/Downloads/nii'  
    directories = find_subject_id(csv_path, root_dir)
    print(len(directories))

find_anomal()

# s1 =set(['027_S_0074', '018_S_2155', '027_S_2219', '094_S_2238', '127_S_1427', '068_S_2315', '041_S_4037', '099_S_4086', '041_S_4143', '021_S_4254', '137_S_4351', '018_S_4399', '137_S_4466', '006_S_4485', '137_S_4482', '137_S_4536', '021_S_4659', '024_S_4674', '127_S_4765', '116_S_4855', '041_S_4974', '024_S_6005', '941_S_6017', '941_S_6052', '024_S_6202', '035_S_6306', '019_S_6315', '035_S_6380', '341_S_6494', '941_S_6496', '116_S_6458', '941_S_6514', '941_S_6546', '100_S_6493', '941_S_6580', '031_S_6715', '016_S_6708', '126_S_6724', '016_S_6800', '168_S_6860', '137_S_6880', '168_S_6902', '016_S_6939', '126_S_7060', '941_S_10013', '021_S_0337', '011_S_10026', '011_S_6303', '035_S_10068', '126_S_4514', '068_S_0127', '052_S_4944', 
# '126_S_0605', '941_S_10212'])
# s2 ={'024_S_6005', '941_S_10212', '137_S_4351', '018_S_2155', '068_S_0127', '137_S_4536', '018_S_4399', '941_S_6017', '941_S_6052', '341_S_6494', '100_S_6493', '116_S_6458', '137_S_4466', '137_S_4482', '041_S_4143', '941_S_6514', '941_S_6496', '016_S_6800', '094_S_2238', '016_S_6939', '116_S_4855', '019_S_6315', '035_S_6380', '168_S_6902', '011_S_10026', '941_S_10013', '041_S_4037', '035_S_6306', '041_S_4974', '024_S_6202', '016_S_6708', '006_S_4485', '941_S_6546', '068_S_2315', '137_S_6880', '031_S_6715', '168_S_6860', '941_S_6580', '024_S_4674', '052_S_4944', '011_S_6303'}

# print(s1 - s2)
