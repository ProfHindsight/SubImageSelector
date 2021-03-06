'''

OK, so what we're doing here is scanning through all of the "BIRD" files, extracting the sub-image index from the filename,
and copying X number of "NOBIRD" files into the output location.

All "BIRD" files will be used. 

'''

BIRD_FOLDER_LOC = "S:\\Thesis_Storage\\Zephyr\\dataset\\bird"
NOBIRD_FOLDER_LOC = "S:\\Thesis_Storage\\Zephyr\\dataset\\nobird"
NOBIRDLIKELY_FOLDER_LOC = "S:\\Thesis_Storage\\Zephyr\\dataset\\nobirdlikely"
OUTPUT_FOLDER_LOC = "S:\\Thesis_Storage\\Zephyr\\dataset\\nobird_matched"

GRAYSCALE_TEST_SUBIMAGE = 13

import re
import numpy as np
import os
import matplotlib.image as mpimg
import random
import shutil

def check_if_grayscale(filepath):
    '''
    This will check the diagnonal for 128 values equal to each other
    '''
    img = mpimg.imread(filepath)
    for i in range(0,128):
        if ((2*img[i,i,1]) != (img[i,i,2] + img[i,i,0])):
            return False
    return True

class CameraLocation:
    '''
    Camera location is unique for each set of 10000 images
    Each of the 10000 may have been separated into up to 176 sub-images
    If there was a bird processed, it was put in BIRD_FOLDER_LOC
    If there was no-bird processed, it was put in NOBIRD_FOLDER_LOC
    If there was no-bird pre-processed, it was put in NOBIRDLIKELY_FOLDER_LOC

    This class is meant to try organize the following information:
    - Which camera location it belongs to 
    - Whether the image is grayscale or not (throw out if it is)
    - Whether the sub-image has already been selected for use
    - Whether the sub-image exists in the nobird or nobirdlikely folder
    - Whether the image is labeled with anything extra - Probably just going to throw it out

    - NOTE: Numpy array axes should be divisible by 8 so we can pack them into the byte array 
            without having to do anything special
    '''
    def __init__(self, number, date, group, special_chars):
        self.number = number
        self.date = date
        self.group = group
        self.special_chars = special_chars
        self.min_image_index = 9999
        self.max_image_index = 0
        self.is_grayscale = np.zeros(10000, dtype=bool) # Maximum number of images per location
        self.is_used = np.zeros((10000, 176), dtype=bool) # Subimage index, image if used
        self.is_nobirdlikely = np.zeros(10000, dtype=bool)
        self.is_bird = np.zeros((10000, 176), dtype=bool)
        self.is_special_name = np.zeros(10000, dtype=bool)
        self.is_present = np.zeros(10000, dtype=bool)
        self.training_data = np.zeros(10000, dtype=bool)

    def mark_grayscale(self, index):
        self.is_grayscale[index] = 1

    def mark_subimage_used(self, image_index, subimage_index):
        self.is_used[image_index, subimage_index] = 1

    def mark_nobirdlikely(self, image_index):
        self.is_nobirdlikely[image_index] = 1

    def mark_bird(self, image_index, subimage_index):
        self.is_bird[image_index, subimage_index] = 1

    def mark_special_name(self, image_index):
        self.is_special_name[image_index] = 1

    def mark_is_present(self, image_index):
        self.is_present[image_index] = 1

    def mark_subimage_for_training(self, image_index):
        self.training_data[image_index] = 1

    def set_special_chars(self, special_chars):
        self.special_chars = special_chars

    def get_grayscale(self, index):
        return self.is_grayscale[index] == 1
    
    def get_used(self, image_index, subimage_index):
        return self.is_used[image_index, subimage_index] == 1

    def get_nobirdlikely(self, image_index):
        return self.is_nobirdlikely[image_index] == 1

    def get_bird(self, image_index, subimage_index):
        return self.is_bird[image_index, subimage_index] == 1

    def get_is_special_name(self, image_index):
        return self.is_special_name[image_index] == 1
    
    def get_present(self, image_index):
        return self.is_present[image_index] == 1
    
    def get_special_chars(self, special_chars):
        return self.special_chars

    def where_located(self, image_index, subimage_index):
        if self.get_bird(image_index, subimage_index):
            return BIRD_FOLDER_LOC
        elif self.get_nobirdlikely(image_index) == 1:
            return NOBIRDLIKELY_FOLDER_LOC
        else :
            return NOBIRD_FOLDER_LOC

    def update_min_count(self, test_num):
        if test_num < self.min_image_index:
            self.min_image_index = test_num

    def update_max_count(self, test_num):
        if test_num > self.max_image_index:
            self.max_image_index = test_num
    
    def match(self, number, date, group):
        return self.number == number and self.date == date and self.group == group

    def generate_filename(self, image_index, subimage_index):
        # Camera 1_2. 5 Oct - 28 Oct 2016_100RECNX_IMG_0927 b_052
        if self.get_nobirdlikely(image_index):
            return f'Camera {self.number}_{self.date}_{self.group}RECNX_IMG_{image_index:04}_{subimage_index:03}.png'
        else:
            return f'Camera {self.number}_{self.date}_{self.group}RECNX_IMG_{image_index:04} b_{subimage_index:03}.png'
        
    def generate_source_image_filepath(self, image_index):
        # Camera 1/1. 28 Sept - 5 Oct 2016/100RECNX/IMG_0055.JPG
        if self.get_nobirdlikely(image_index):
            return f'Camera {self.number}/{self.date}/{self.group}RECNX/IMG{image_index:04}.JPG'
        else:
            return f'Camera {self.number}/{self.date}/{self.group}RECNX/IMG{image_index:04} b.JPG'

    def add_subimage_info(self, source, image_index, subimage_index):
        image_index_n = int(image_index)
        subimage_index_n = int(subimage_index)
        self.mark_is_present(image_index_n)
        self.update_max_count(image_index_n)
        self.update_min_count(image_index_n)
        if source == "BIRD":
            self.mark_bird(image_index_n, subimage_index_n)
            self.mark_subimage_used(image_index_n, subimage_index_n)
            self.mark_subimage_for_training(image_index_n)
        elif source == "NOBIRDLIKELY":
            self.mark_nobirdlikely(image_index_n)
        if subimage_index_n == GRAYSCALE_TEST_SUBIMAGE:
            filepath = os.path.join(self.where_located(image_index_n, subimage_index_n), self.generate_filename(image_index_n, subimage_index_n))
            if check_if_grayscale(filepath):
                self.is_grayscale[image_index_n] = 1

    def generate_unused_nobird_filepath(self, subimage_index):
        '''
        returns the source folder location and file name in a tuple
        subimage_index - the subimage we are targeting
        '''
        incrementing_numbers = np.array(range(0, self.is_grayscale.size))
        
        '''
        1. Making sure the sub-image isn't grayscale
        2. Making sure the image file is actually present
        3. Making sure the sub-image hasn't already been selected for use
        '''
        valid_locations = \
            ~self.is_grayscale & \
            self.is_present & \
            ~self.is_used[:, subimage_index]
        
        '''
        This is an addition (which probably warrants a re-write) to minimize the number of "nobird"
        images used for training. The purpose of this is to use as few files as possible for training
        so the rest can be used in the overall dataset evaluation after the network is trained.
        '''
        if (np.sum(valid_locations & self.training_data) == 0):
            unused_locations = valid_locations & ~self.training_data
            unused_indicies = incrementing_numbers[unused_locations]
            unused_index = unused_indicies[random.randint(0, unused_indicies.size-1)]
            self.mark_subimage_for_training(unused_index)
        
        valid_locations = valid_locations & self.training_data

        if np.sum(valid_locations) == 0:
            print(f'No valid indicies!')
            return ("","")
        valid_indicies = incrementing_numbers[valid_locations]
        image_index = valid_indicies[random.randint(0, valid_indicies.size-1)]
        filename = self.generate_filename(image_index, subimage_index)
        self.mark_subimage_used(image_index, subimage_index)
        return (self.where_located(image_index, subimage_index), filename)

    
class CameraLocationArray:
    def __init__(self):
        self.cameras = [CameraLocation(0,0,0,0)]
        self.last_index = 0
        self.first_time = True
    
    def add_camera(self, camera):
        self.cameras.append(camera)

    def get_camera(self, number, date, group, special_chars):
        if self.cameras[self.last_index].match(number, date, group):
            return self.cameras[self.last_index]
        else:
            for camera in self.cameras:
                if camera.match(number, date, group):
                    return camera
            self.add_camera(CameraLocation(number, date, group, special_chars))
            return self.cameras[-1]


# OPERATIONAL VARIABLES or something
TEST = 1
# Camera, Date, Group, Image Index, Special Characters, Subimage Index
regex_string = r'Camera ([0-6])_([0-9A-Za-z -.]*)_([0-3]*)RECNX_IMG_([0-9]*)([ a-zA-Z_]*)_([0-9]*).png'

# -------------------------------------------------------
#           GATHER THE DATA
# -------------------------------------------------------
def generate_metadata():
    CLA = CameraLocationArray()

    # BIRD_FOLDER_LOCATION Information
    for filename in os.scandir(BIRD_FOLDER_LOC):
        match_obj = re.match(regex_string, filename.name)
        (number, date, group, image_index, special_chars, subimage_index) = match_obj.groups()
        camera = CLA.get_camera(number, date, group, special_chars)
        if special_chars != " b":
            camera.mark_special_name(int(image_index))
        else:
            camera.add_subimage_info("BIRD", image_index, subimage_index)

    # NOBIRD_FOLDER_LOC Information
    for filename in os.scandir(NOBIRD_FOLDER_LOC):
        match_obj = re.match(regex_string, filename.name)
        (number, date, group, image_index, special_chars, subimage_index) = match_obj.groups()
        camera = CLA.get_camera(number, date, group, special_chars)
        if special_chars != " b":
            camera.mark_special_name(int(image_index))
        else:
            camera.add_subimage_info("NOBIRD", image_index, subimage_index)

    # NOBIRDLIKELY_FOLDER_LOC Information
    for filename in os.scandir(NOBIRDLIKELY_FOLDER_LOC):
        match_obj = re.match(regex_string, filename.name)
        (number, date, group, image_index, special_chars, subimage_index) = match_obj.groups()
        camera = CLA.get_camera(number, date, group, special_chars)
        if special_chars != "":
            camera.mark_special_name(int(image_index))
        else:
            camera.add_subimage_info("NOBIRDLIKELY", image_index, subimage_index)

    print("Number of files for each camera location")
    for camera in CLA.cameras:
        print(f'{camera.min_image_index}:{camera.max_image_index}')

    # Omit the first one since it's just zeros
    if CLA.cameras[0].date == 0:
        CLA.cameras = CLA.cameras[1:]

    return CLA

# -------------------------------------------------------
#           GENERATE THE MATCHED DATASET
# -------------------------------------------------------
NOBIRD_FILE_MULTIPLIER = 5
def generate_matched_dataset(CLA):
    for filename in os.scandir(BIRD_FOLDER_LOC):
        match_obj = re.match(regex_string, filename.name)
        (number, date, group, _, spec_char, subimage_index) = match_obj.groups()
        camera = CLA.get_camera(number, date, group, spec_char)
        for _ in range(0, NOBIRD_FILE_MULTIPLIER):
            (src_folder, src_name) = camera.generate_unused_nobird_filepath(int(subimage_index))
            src = os.path.join(src_folder, src_name)
            dest = os.path.join(OUTPUT_FOLDER_LOC, src_name)
            shutil.copy2(src, dest)

        if TEST == 1:
            break
    
    output_image_used_csv = os.path.join(OUTPUT_FOLDER_LOC, "used_images.csv")
    with open(output_image_used_csv, 'w') as csv_file:
        csv_file.write("Filename\n")
        for camera in CLA.cameras:
            for i in range(0, len(camera.is_grayscale)):
                if camera.training_data[i] == True:
                    csv_file.write(camera.generate_source_image_filepath(i) + "\n")

    print("Matched Dataset Generated")


# -------------------------------------------------------
#           SPLIT MATCHED DATASET
# -------------------------------------------------------
DATASET_FOLDER_LOC = "S:\\Thesis_Storage\\Zephyr\\mini-dataset-matched"
def split_matched_dataset(train_percent):
    if train_percent > 1 or train_percent < 0:
        print("Please provide a floating point number bewteen 0 and 1")
        return

    # Split the bird dataset
    for filename in os.scandir(BIRD_FOLDER_LOC):
        src = os.path.join(BIRD_FOLDER_LOC, filename.name)
        if random.random() < train_percent:
            dest = os.path.join(DATASET_FOLDER_LOC, "train\\bird\\")
            shutil.copy2(src, dest)
        else:
            dest = os.path.join(DATASET_FOLDER_LOC, "val\\bird\\")
            shutil.copy2(src, dest)

    # Split the no-bird dataset
    for filename in os.scandir(OUTPUT_FOLDER_LOC):
        src = os.path.join(OUTPUT_FOLDER_LOC, filename.name)
        if random.random() < train_percent:
            dest = os.path.join(DATASET_FOLDER_LOC, "train\\nobird\\")
            shutil.copy2(src, dest)
        else:
            dest = os.path.join(DATASET_FOLDER_LOC, "val\\nobird\\")
            shutil.copy2(src, dest)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER_LOC):
        os.mkdir(OUTPUT_FOLDER_LOC)

    if not os.path.exists(DATASET_FOLDER_LOC):
        os.mkdir(DATASET_FOLDER_LOC)
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "train"))
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "train\\bird"))
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "train\\nobird"))
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "val"))
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "val\\bird"))
        os.mkdir(os.path.join(DATASET_FOLDER_LOC, "val\\nobird"))
        
    CLA = generate_metadata()
    generate_matched_dataset(CLA)
    split_matched_dataset(0.8)