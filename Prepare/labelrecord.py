import pandas as pd 
import pydicom

# Simple class to manage label records.
class LabelRecord(object):

    def __init__(self, filename='', hasBoundingBox=False, boundingBoxes=[]):
        self.filename = filename
        self.hasBoundingBox = hasBoundingBox
        self.boundingBoxes = boundingBoxes

    def load(self, label_file):
        extract = lambda row: [row['y'], row['x'], row['height'], row['width']]
        data = pd.read_csv(label_file)
        records = {}
        for _, row in data.iterrows():
            pid = row['patientId']
            if pid not in records:
                records[pid] = LabelRecord('%s.dcm' % pid, True if row['Target'] == 1 else False)
            if records[pid].hasBoundingBox:
                records[pid].boundingBoxes.append(extract(row))
        
        return records
           