# Defect-Detection-in-Solar-Cells
The model-training files for our Master thesis: Defect Detection in Solar Cells: Leveraging Deep-Learning Technology. The files currently uses the ELPV-dataset and the PVEL-AD dataset for it's defect detection tasks. It contains the run files as well as files containing different functions used in those run files.

# Instructions
The run-files (run.py and CAM.py) currently expect the Data to contain the images and labels from the PVEL-AD and the ELPV datasets as they are provided. As such, drop the PVEL-AD and the ELPV folders directly into the Data folder. Parts of the run files requires functions provided by the datasets (ELPV's elpv_reader.py).

# Requirements
Requires grad_cam_plus function from gradcam.py in https://github.com/samson6460/tf_keras_gradcamplusplus

# Links

The ELPV dataset and it's functions can be found at: https://github.com/zae-bayern/elpv-dataset <br />
The PVEL-AD dataset is not publically available, but you can request access to the dataset by following the instructions at: https://github.com/binyisu/PVEL-AD

# Authors
Vegard Helland <br />
Martin Johansen
