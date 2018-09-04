import os
import numpy as np
import uuid
import pydicom

from PIL import Image

# converts the raw pixel data into [h,w,c] ndarray
def make_image_data(pixel_array):
    return np.stack([pixel_array] * 3, axis = 2)

# extracts the given bounding box from the [h,w,c] ndarray for the image
def extract_image(image_data, box):
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width
    return image_data[y1:y2,x1:x2]

# writes the extracted bounding box to the given output path with new filename
def write_image(image_data, path):
    im = Image.fromarray(image_data)
    im.save(os.path.join(path,'%s.jpg' % str(uuid.uuid4())))
    
# gets the image data from the dicom file
def get_image_data(filename, path):
    fullpath = os.path.join(path, filename)
    dcm_data = pydicom.read_file(fullpath)
    return make_image_data(dcm_data.pixel_array)
