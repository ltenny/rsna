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

    # computes the direction and energy for the given pixel centered in a 3x3 matrix of the data 
    # using the right singular vector and singular values from the SVD of the 9x2 matrix of the gradients, energy is computed using
    # R = (s1 - s2)/(s1 + s2) for stability, returns [direction, energy, 0]
    # NB: grads has shape [2] where [0] is partial df/dy and [1] is partial df/dx, this yields a 9x2 matrix, partials df/dx
    # in first col, partials df/dy in second column, 3*3=9 rows, hence 9x2, called F, svd(F) = U*S*V, we use S for energy
    # and V[:,0] for direction
    @staticmethod
    def compute_direction_and_energy(grads, x, y):
        pdy = grads[0][y-1:y+2,x-1:x+2]
        pdx = grads[1][y-1:y+2,x-1:x+2]
        F = np.column_stack((np.reshape(pdx,[-1]),np.reshape(pdy,[-1]))) # F is 9x2 [[pdx, pdy]...[pdx, pdy]]
        _,S,V = np.linalg.svd(F)
        if S[0] != S[0] or S[1] != S[1] or (S[1] + S[0]) == 0.:
            energy = 0.0
        else:
            energy = (S[0] - S[1])/(S[0] + S[1])
        rotateTransform = np.array([[0., -1.],[1.,0.]]) # need normal to gradient, so rotate by pi/2
        direction = V[:,0].dot(rotateTransform)
        return direction, energy

    @staticmethod 
    def xlate_image(image_data):
        h = image_data.shape[0]
        w = image_data.shape[1]
        one_channel = image_data[:,:,0:1]
        reshaped = np.reshape(one_channel,[h,-1])
        grads = np.gradient(reshaped)
        result = np.full((h,w,3),0.0)
        for r in range(1,h-1,1):
            for c in range(1,w-1,1):
                d,e = CXRImage.compute_direction_and_energy(grads,c,r)
                result[r][c][0] = d[0]
                result[r][c][1] = d[1]
                if e != e:
                    e = 0.0
                result[r][c][2] = e
        delta = abs(np.min(result))
        divby = np.max(result+delta)
        return ((result + delta)/divby)


    @staticmethod
    def extract_anisotropic_scale_and_write(image_data, box, height, width, path, filename=None):
        h = image_data.shape[0]
        w = image_data.shape[1]
        extracted_image = CXRImage.extract_image(image_data,box)
        rgb_average = np.array([np.average(extracted_image[:,:,i]) for i in range(3)]).astype(int)
        new_image = np.full((h, w, 3), rgb_average,dtype=np.uint8)
        bw = box[2] - box[0] # box width
        bh = box[3] - box[1] # box height
        y1 = np.ceil(h/2).astype(int) - np.floor(bh/2).astype(int)
        x1 = np.ceil(w/2).astype(int) - np.floor(bw/2).astype(int)
        new_image[y1:y1+bh,x1:x1+bw] = extracted_image
        y1 = np.ceil(h/2).astype(int) - np.floor(height/2).astype(int)
        x1 = np.ceil(w/2).astype(int) - np.floor(width/2).astype(int)
        cropped_image = new_image[y1:y1+height,x1:x1+width]
        
        im = Image.fromarray(cropped_image)
        im.thumbnail((height, width), Image.ANTIALIAS)
        if filename:
            im.save(os.path.join(path, filename))
        else:
            im.save(os.path.join(path,'%s.jpg' % str(uuid.uuid4())))

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


