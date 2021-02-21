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
    def __init__(self, number, date, group):
        self.number = number
        self.date = date
        self.group = group
        self.min_image_index = 9999
        self.max_image_index = 0
        self.is_grayscale = np.zeros(10000, dtype=bool) # Maximum number of images per location
        self.is_used = np.zeros((10000, 176), dtype=bool) # Subimage index, image if used
        self.is_nobirdlikely = np.zeros(10000, dtype=bool)
        self.is_bird = np.zeros((10000, 176), dtype=bool)
        self.special_name = np.zeros(10000, dtype=bool)
        self.is_present = np.zeros(10000, dtype=bool)

    ## Don't need this as this time - remove on a future commit
    # def to_dict(self):
    #     camdict = {
    #         "number"                : self.number,
    #         "date"                  : self.date,
    #         "group"                 : self.group,
    #         "min_index"             : self.min_image_index,
    #         "max_index"             : self.max_image_index,
    #         "grayscale"             : np.packbits(self.is_grayscale).tobytes(),
    #         "is_used"               : np.packbits(self.is_used).tobytes(),
    #         "is_nobirdlikely"       : np.packbits(self.is_nobirdlikely).tobytes(),
    #         "is_bird"               : np.packbits(self.is_bird).tobytes(),
    #         "special_name"          : np.packbits(self.special_name).tobytes(),
    #         "is_present"            : np.packbits(self.is_present).tobytes(),
    #         "grayscale_rows"        : self.is_grayscale.shape[0],
    #         "grayscale_cols"        : self.is_grayscale.shape[1],
    #         "is_used_rows"          : self.is_used.shape[0],
    #         "is_used_cols"          : self.is_used.shape[1],
    #         "is_nobirdlikely_rows"  : self.is_nobirdlikely.shape[0],
    #         "is_nobirdlikely_cols"  : self.is_nobirdlikely.shape[1],
    #         "is_bird_rows"          : self.is_bird.shape[0],
    #         "is_bird_cols"          : self.is_bird.shape[1],
    #         "special_name_rows"     : self.special_name.shape[0],
    #         "special_name_cols"     : self.special_name.shape[1],
    #         "is_present_rows"       : self.is_present.shape[0],
    #         "is_present_cols"       : self.is_present.shape[1] 
    #     }
    #     return camdict
    
    # def load_from_dict(self, camdict):
    #     def unpack_byte_array(bytear, rows, cols):
    #         array = np.frombuffer(bytear, dtype=np.uint8)
    #         array = np.unpackbits(array)
    #         array = np.reshape(array, (int(rows), int(cols)))
    #         return np.array(array, dtype=bool)
    #     self.number                 = camdict["number"]
    #     self.date                   = camdict["date"]
    #     self.group                  = camdict["group"]
    #     self.min_image_index        = camdict["min_index"]
    #     self.max_image_index        = camdict["max_index"]
    #     self.grayscale              = unpack_byte_array(camdict["grayscale"], camdict["grayscale_rows"], camdict["grayscale_cols"])
    #     self.is_used                = unpack_byte_array(camdict["is_used"], camdict["is_used_rows"], camdict["is_used_cols"])
    #     self.is_nobirdlikely        = unpack_byte_array(camdict["is_nobirdlikely"], camdict["is_nobirdlikely_rows"], camdict["is_nobirdlikely_cols"])
    #     self.is_bird                = unpack_byte_array(camdict["is_bird"], camdict["is_bird_rows"], camdict["is_bird_cols"])
    #     self.special_name           = unpack_byte_array(camdict["special_name"], camdict["special_name_rows"], camdict["special_name_cols"])
    #     self.is_present             = unpack_byte_array(camdict["is_present"], camdict["is_present_rows"], camdict["is_present_cols"])

    def mark_grayscale(self, index):
        self.is_grayscale[index] = 1

    def mark_subimage_used(self, image_index, subimage_index):
        self.is_used[image_index, subimage_index] = 1

    def mark_nobirdlikely(self, image_index):
        self.is_nobirdlikely[image_index] = 1

    def mark_bird(self, image_index, subimage_index):
        self.is_bird[image_index, subimage_index] = 1

    def mark_special_name(self, image_index):
        self.special_name[image_index] = 1

    def mark_is_present(self, image_index):
        self.is_present[image_index] = 1

    def get_grayscale(self, index):
        return self.is_grayscale[index] == 1
    
    def get_used(self, image_index, subimage_index):
        return self.is_used[image_index, subimage_index] == 1

    def get_nobirdlikely(self, image_index):
        return self.is_nobirdlikely[image_index] == 1

    def get_bird(self, image_index, subimage_index):
        return self.is_bird[image_index, subimage_index] == 1

    def get_special_name(self, image_index):
        return self.is_special_name[image_index] == 1
    
    def get_present(self, image_index):
        return self.is_present[image_index] == 1

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
        if self.get_nobirdlikely(image_index):
            return f'Camera {self.number}_{self.date}_{self.group}RECNX_IMG_{image_index:04}_{subimage_index:03}.png'
        else:
            return f'Camera {self.number}_{self.date}_{self.group}RECNX_IMG_{image_index:04} b_{subimage_index:03}.png'

    def add_subimage_info(self, source, image_index, subimage_index):
        image_index_n = int(image_index)
        subimage_index_n = int(subimage_index)
        camera.mark_is_present(image_index_n)
        camera.update_max_count(image_index_n)
        camera.update_min_count(image_index_n)
        if source == "BIRD":
            camera.mark_bird(image_index_n, subimage_index_n)
            camera.mark_subimage_used(image_index_n, subimage_index_n)
        elif source == "NOBIRDLIKELY":
            camera.mark_nobirdlikely(image_index_n)
        if subimage_index_n == GRAYSCALE_TEST_SUBIMAGE:
            filepath = os.path.join(self.where_located(image_index_n, subimage_index_n), self.generate_filename(image_index_n, subimage_index_n))
            if check_if_grayscale(filepath):
                self.is_grayscale[image_index_n] = 1

    def generate_unused_filepath(self, subimage_index):
        '''
        returns the source folder location and file name in a tuple
        '''
        incrementing_numbers = np.array(range(0, self.is_grayscale.size))
        valid_locations = \
            ~self.is_grayscale & \
            self.is_present & \
            ~self.is_used[:, subimage_index]
        if np.sum(valid_locations) == 0:
            print(f'No valid indicies!')
            return ""
        valid_indicies = incrementing_numbers[valid_locations]
        image_index = valid_indicies[random.randint(0, valid_indicies.size-1)]
        filename = self.generate_filename(image_index, subimage_index)
        self.mark_subimage_used(image_index, subimage_index)
        return (self.where_located(image_index, subimage_index), filename)

    
class CameraLocationArray:
    def __init__(self):
        self.cameras = [CameraLocation(0,0,0)]
        self.last_index = 0
        self.first_time = True
    
    def add_camera(self, camera):
        self.cameras.append(camera)

    def get_camera(self, number, date, group):
        if self.cameras[self.last_index].match(number, date, group):
            return self.cameras[self.last_index]
        else:
            for camera in self.cameras:
                if camera.match(number, date, group):
                    return camera
            self.add_camera(CameraLocation(number, date, group))
            return self.cameras[-1]

CLA = CameraLocationArray()

# Camera, Date, Group, Image Index, Special Characters, Subimage Index
regex_string = r'Camera ([0-6])_([0-9A-Za-z -.]*)_([0-3]*)RECNX_IMG_([0-9]*)([ a-zA-Z_]*)_([0-9]*).png'

for filename in os.scandir(BIRD_FOLDER_LOC):
    match_obj = re.match(regex_string, filename.name)
    (number, date, group, image_index, spec_char, subimage_index) = match_obj.groups()
    camera = CLA.get_camera(number, date, group)
    if spec_char != " b":
        camera.mark_special_name(int(image_index))
    else:
        camera.add_subimage_info("BIRD", image_index, subimage_index)

# Commenting this out to not use bird images. The first sample images had birds in them when they shouldn't have.
# for filename in os.scandir(NOBIRD_FOLDER_LOC):
#     match_obj = re.match(regex_string, filename.name)
#     (number, date, group, image_index, spec_char, subimage_index) = match_obj.groups()
#     camera = CLA.get_camera(number, date, group)
#     if spec_char != " b":
#         camera.mark_special_name(int(image_index))
#     else:
#         camera.add_subimage_info("NOBIRD", image_index, subimage_index)

for filename in os.scandir(NOBIRDLIKELY_FOLDER_LOC):
    match_obj = re.match(regex_string, filename.name)
    (number, date, group, image_index, spec_char, subimage_index) = match_obj.groups()
    camera = CLA.get_camera(number, date, group)
    if spec_char != "":
        camera.mark_special_name(int(image_index))
    else:
        camera.add_subimage_info("NOBIRDLIKELY", image_index, subimage_index)

# Omit the first one since it's just zeros
if CLA.cameras[0].date == 0:
    CLA.cameras = CLA.cameras[1:]

# Iterate through the bird files and find matching no-bird files.
NOBIRD_FILE_MULTIPLIER = 5

for filename in os.scandir(BIRD_FOLDER_LOC):
    match_obj = re.match(regex_string, filename.name)
    (number, date, group, image_index, spec_char, subimage_index) = match_obj.groups()
    camera = CLA.get_camera(number, date, group)
    for i in range(0, NOBIRD_FILE_MULTIPLIER):
        (src_folder, src_name) = camera.generate_unused_filepath(int(subimage_index))
        src = os.path.join(src_folder, src_name)
        dest = os.path.join(OUTPUT_FOLDER_LOC, src_name)
        shutil.copy2(src, dest)
    
print("Number of files for each camera location")
for camera in CLA.cameras:
    print(f'{camera.min_image_index}:{camera.max_image_index}')