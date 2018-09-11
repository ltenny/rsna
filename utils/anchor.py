from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 

class Anchor(object):

    def __init__(self, base_size = 16, ratios=[0.5,1,2], scales=2**np.arange(3,6)):
        self._base_size = base_size
        self._ratios = ratios
        self._scales = scales
        self._anchors = []

    def generate(self):
        base_anchor = np.array([1,1,self._base_size, self._base_size]) - 1
        ratio_anchors = self._ratio_enum(base_anchor, self._ratios)
        self._anchors = np.vstack([self._scale_enum(ratio_anchors[i,:], self._scales) for i in range(ratio_anchors.shape[0])])
        return self._anchors

    def _coords(self, anchor):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_center = anchor[0] + 0.5 * (w - 1)
        y_center = anchor[1] + 0.5 * (h - 1)
        return w, h, x_center, y_center

    def _ratio_enum(self, anchor, ratios):
        w, h, x_center, y_center = self._coords(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(np.sqrt(ws * ratios))
        return self._make_anchors(ws,hs,x_center, y_center)

    def _make_anchors(self, ws, hs, x_center, y_center):
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        return np.hstack((x_center - 0.5 * (ws - 1),
                          y_center - 0.5 * (hs - 1),
                          x_center + 0.5 * (ws - 1),
                          y_center + 0.5 * (hs - 1)))
    
    def _scale_enum(self, anchor, scales):
        w, h, x_center, y_center = self._coords(anchor)
        ws = w * scales
        hs = h * scales
        return self._make_anchors(ws, hs, x_center, y_center)
    