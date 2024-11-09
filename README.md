#Abstract:

Ghost-HRNet is introduced, a lightweight human pose estimation network that combines the efficient feature extraction capabilities of the Ghost module with the multi-scale feature fusion advantages of HRNet. By introducing depthwise separable convolution and convolutional block attention module (CBAM), Ghost-HRNet significantly reduces the number of parameters and computational burden while maintaining high accuracy, making it very suitable for real-time applications. Experimental results show that Ghost-HRNet achieves average accuracy of 66% and 87.26% on the COCO and MPII data sets respectively, while reducing parameter size by 71.3% and computing load by 79.0%.


#Requirement:

Python 3.7 或 3.8
PyTorch >= v1.13.1
NumPy >= v1.21.6
Pandas >= v1.4.2
Matplotlib >= v3.5.2
Scikit-learn >= v1.1.1
OpenCV-Python >= v4.6.0.66
tqdm >= v4.64.0
Pillow >= v9.2.0
PyYAML >= v6.0
TensorBoard >= v2.10.1

#How to use:

The project contains the following main folders: experiments, lib, config, and model. The experiments folder contains two subfolders, coco and mpii, corresponding to two datasets respectively. Each subfolder contains the weight files of the proposed model and the baseline model. The lib folder contains the main code to implement the model, the config folder provides the basic configuration file, and the model folder contains the specific code implementation of each model.

Before using this code, you need to configure the environment first to ensure that all dependent libraries are installed. It is recommended to quickly install the required Python libraries according to the information in requirements.

When training the model, you need to find the configuration file of the corresponding dataset in the config folder (such as coco_config.yaml or mpii_config.yaml), and adjust the configuration parameters as needed, such as the dataset path, batch size, learning rate, and number of training rounds. Run the train.py script to start training. After training, you can use the generated weight file for testing. Run test.py and specify the path of the configuration file and weight file. Make sure that the weight file path is consistent with the corresponding weight file in the experiments folder. After the test is completed, the model performance indicators will be output to the terminal or stored in the specified path.


