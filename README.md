# Cropyieldprediction course assignment

File Description
train.py: Model training and validation code. Note: This file does not need to be run for this assignment as we only provide test data.
test.py: Model testing code. This is the main file to run for testing.
dataload.py: Data loading utilities.
augmentation.py: Data augmentation procedures.
ConvLSTM.py: Model architecture implementation.
checkpoint.pth: Pre-trained model weights (required for testing).
showimage.py: Image visualization utility. The provided county-level remote sensing data is normalized and contains 10 bands stored in .npyformat, which cannot be directly visualized. This script extracts RGB bands and converts them to viewable JPG format.



Usage Instructions
Step 1: Environment Setup
Please refer to the PDF documentation for detailed environment configuration steps.
Step 2: Data Preparation
The data/folder contains sample datasets for initial testing. Use these to ensure the network runs properly.
Download the complete test set from: Google Drive Link
Place the downloaded data in the data/folder.
Step 3: Model Testing
Please follow the testing procedure outlined in the PDF documentation.


Notes
This project focuses on testing pre-trained models rather than training new ones.
Ensure all dependencies are installed before running the code.
The visualization script (showimage.py) is provided to help inspect the remote sensing data.
