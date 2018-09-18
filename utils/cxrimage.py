import os
import numpy as np
import uuid
import pydicom

from PIL import Image

class CXRImage(object):
    # converts the raw pixel data into [h,w,c] ndarray
    @staticmethod
    def make_image_data(pixel_array):
        return np.stack([pixel_array] * 3, axis = 2)

    # extracts the given bounding box from the [h,w,c] ndarray for the image
    @staticmethod
    def extract_image(image_data, box):
        x1,y1,x2,y2 = box
        return image_data[y1:y2,x1:x2]

    # writes the extracted bounding box to the given output path with new filename
    @staticmethod
    def write_image(image_data, path, filename=None):
        im = Image.fromarray(image_data)
        if filename:
            im.save(os.path.join(path, filename))
        else:
            im.save(os.path.join(path,'%s.jpg' % str(uuid.uuid4())))

    @staticmethod
    def overlay_box(im, box, rgb, stroke=6):
        x1,y1,x2,y2 = box
        im[y1:y1+stroke, x1:x2] = rgb
        im[y2:y2+stroke, x1:x2] = rgb
        im[y1:y2, x1:x1+stroke] = rgb
        im[y1:y2, x2:x2+stroke] = rgb
        return im    

    @staticmethod
    def write_image_with_bounding_boxes(image_data, path, filename, boxes, rgb=[128,0,0]):
        for i in range(0,boxes.shape[0]):
            box = boxes[i,:]
            image_data = CXRImage.overlay_box(image_data, box, rgb)
        im = Image.fromarray(image_data)
        im.save(os.path.join(path, filename))

    # gets the image data from the dicom file
    @staticmethod
    def get_image_data(filename, path):
        fullpath = os.path.join(path, filename)
        dcm_data = pydicom.read_file(fullpath)
        return CXRImage.make_image_data(dcm_data.pixel_array)

    # extracts box from image, centers on new image with background set to
    # the average of the extracted box, writes the new image to the given file
    @staticmethod
    def extract_center_and_write(image_data, box, height, width, path, filename=None) :
        extracted_image = CXRImage.extract_image(image_data,box)
        rgb_average = np.array([np.average(extracted_image[:,:,i]) for i in range(3)]).astype(int)
        new_image = np.full((height, width,3), rgb_average,dtype=np.uint8)
        bw = box[2] - box[0] # box width
        bh = box[3] - box[1] # box height
        y1 = np.ceil(height/2).astype(int) - np.floor(bh/2).astype(int)
        x1 = np.ceil(width/2).astype(int) - np.floor(bw/2).astype(int)
        new_image[y1:y1+bh,x1:x1+bw] = extracted_image
        CXRImage.write_image(new_image, path, filename)


