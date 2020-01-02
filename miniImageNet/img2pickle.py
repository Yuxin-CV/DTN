import pickle
from PIL import Image
import os
import numpy as np
import random

def main(split):
    train_folder = './' + split
    

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]

    train_data = {}
    labels = []
    data = []
    idx = 0
    for folder in metatrain_folders:
        print(folder)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img = np.asarray(Image.open(img_path).convert('RGB').resize((224,224)))
            data.append(img)
            labels.append(idx)
        idx = idx+1

    train_data['labels'] = labels
    train_data['data'] = np.array(data)

    save_pickle(train_data, 'miniImageNet_category_split_'+split+'.pickle')
    print('save data success', split)


def save_pickle(data, file_name):
    with open(file_name, 'wb') as fo:
        pickle.dump(data, fo, protocol=4)


if __name__ == '__main__':
    
    print('img2pickle start...')

    main('train_val')
    main('test')

    print('img2pickle finished!')
