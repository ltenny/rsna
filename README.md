# rsna
Kaggle RSNA Pneumonia Detection Challenge

Notes
-----
- the utils directory contains various helpful utilties and unit tests
- prepare.py takes the original .dcm files and creates three directories
    - positives - which contains images created by extracting ground-true-boxes 
        (GTB) from each CXR that contains one or more GTB and inserts the pixels
        from this GTB centered onto a 1024x1024 background whose RGB is the average
        RGB from the GTB pixels

    - negatives - which contains images like those in the positives directory but
        these GTBs are taken from a random selection of GTBs and 'normal' CXRs, these
        should represent 'normal' cooresponding GTBs

    - originals - which contains the original CXRs with all associated GTBs overlayed,
        this directory is used only for information and not in any subsequent processing

    - a TFRecord files is created that contains all images from positives and negatives with
        labels 1 and 0, respectively.
    
    - prepare.py should be run first to create the TFRecord which is used in pre-training

- verify.py takes the TFRecord file created by prepare.py and performs all of the image pre-processing
    that is used in the pre-training for one batch, this is used simply to smoke test the 
    pre-processing to verify