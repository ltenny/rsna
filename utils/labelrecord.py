import pandas as pd 
import pydicom
import numpy as np 

# Simple class to manage label records.
class LabelRecord(object):

    def __init__(self, filename='', hasBoundingBox=False):
        self.filename = filename
        self.hasBoundingBox = hasBoundingBox
        self.boundingBoxes = np.array([],np.int32)

    # box is represented as [x1,y1,x2,y2] which is an ndarray
    def _extract_box(self, row):
        x1 = int(row['x'])
        y1 = int(row['y'])
        x2 = x1 + int(row['width'])
        y2 = y1 + int(row['height'])
        return np.array([x1,y1,x2,y2], np.int32)

    def load(self, label_file):
        data = pd.read_csv(label_file)
        records = {}
        for _, row in data.iterrows():
            pid = row['patientId']
            if pid not in records:
                records[pid] = LabelRecord('%s.dcm' % pid, True if row['Target'] == 1 else False)
            if records[pid].hasBoundingBox:
                if len(records[pid].boundingBoxes) == 0:
                    records[pid].boundingBoxes = np.array([self._extract_box(row)])
                else:
                    records[pid].boundingBoxes = np.vstack([records[pid].boundingBoxes, self._extract_box(row)])

        return records
           