from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 

class Box(object):
    # boxes, both ground truth and proposed regions, for an image are represented and a ndarray
    # as in [[x1,y1,x2,y2...x1,y1,x2,y2]...]

    # clip all boxes so that x1,y1 >=0 and x2 < to_width and y2 < to_height
    @staticmethod
    def clip_boxes(boxes, to_width, to_height):
        boxes[:,0::4] = np.maximum(boxes[:,0::4],0)
        boxes[:,1::4] = np.maximum(boxes[:,1::4],0)
        boxes[:,2::4] = np.minimum(boxes[:,2::4],to_width - 1)
        boxes[:,3::4] = np.minimum(boxes[:,3::4],to_height - 1)
        return boxes