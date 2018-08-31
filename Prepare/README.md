This simple app extracts the ground truth bounding boxes from the CXR images 
and stores them in the given 'positives' directory. Likewise, it extracts a selection
of identical bounding boxes (same extents) from non-labeled CXR images. These are stored
in the 'negatives' directory.
 
The images in the 'positives' and 'negatives' directory are used to train the CNN used to
discriminate between regions with/without suffficent opacity.

Run this app to create the 'positives' and 'negatives' images. 

Use: python main.py --positives_dir=positives --negatives_dir=negatives --label_file=labels.csv --image_path=images

Note that the parameters default to reasonable this. Check out the main.py file for these defaults
