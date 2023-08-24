import os
import pickle

data_dir = '/iris/u/jyang27/asdf/gnm_dataset/cory_hall/cory5_aug13_00_0058'
traj_file = os.path.join(data_dir, 'traj_data.pkl')

with open(traj_file, 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()


def get_folder_names(data_dir):
    folder_names = [
        f
        for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f))
    ]
    return folder_names

folder_names = get_folder_names('/iris/u/jyang27/asdf/gnm_dataset/cory_hall')
print(folder_names)
