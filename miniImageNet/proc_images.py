from __future__ import print_function
import csv
import glob
import os

from PIL import Image

# path_to_images = '../images/'

# Put in correct directory
for datatype in ['train_val', 'test']:
    os.system('mkdir ' + datatype)
    print('Creating ' + datatype + ' directory...')
    with open(datatype + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                print('Creating ' + cur_dir + ' directory...')
                last_label = label
            os.system('cp images/' + image_name + ' ' + cur_dir)
