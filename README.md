# Detection-and-Classification-of-Alzheimers-Disease
The purpose of this paper is to detect Alzheimer’s Disease using Deep Learning and Machine Learning algorithms on the early basis which is being further optimized using CSA(Crow Search Algorithm). Alzheimer’s is one of kind and fatal. The early detection of Alzheimer’s Disease because of it’s progressive risk and patients all around the world. Early detection of AD is promising as it can help lot of patients to predetermine the condition they may face in future. AD being progressive, can be prevented if detected early. On worse stage, the curing of this disease is very difficult and expensive. So, by analyzing the consequences of AD, we can make use of Artificial intelligence technology by using MRI scanned images to classify the patients if they may or may not have AD in future. Using of Bio-inspired algorithm can maximize the result and accuracy for this purpose. After comparing the results of the various AI technologies, CSA came to be the best approach using it with ML algorithms. 

# Proposed Methodology:
1.	Data Acquisition (Datasets) :-  The MRI scan images datasets were used from the ADNI(Alzheimer’s Disease Neuroimaging Initiative) who provide these data for research work. The dataset units comprised of .nii format (Neuroimaging Informatics Technology Initiative). This MRI data set consists typically of various combination of images representing the projection of an anatomical volume onto an image plane (projection or planar imaging), a series of images representing thin slices through a volume (tomographic or multi-slice two-dimensional imaging).

2.	Data pre-processing :-  Optimal slices were selected to provide us with the details on maximum information about the subjects’ brains i.e. the middle most slices along the y-direction. The images were converted into a single channeled image(Gray-Scale).

# Methods:
1)	CNN for MRI classification :-
The CNN used consisted of two 2D convolutional layers having 6 filters of dimension 5x5x5 kernel size of each filter in the first conv layer, 16 filters of dimension 5×5×5 in the second conv layer, four fully-connected (linear) layers (tensor-width * tensor-height *16 hidden nodes in first hidden layer, second hidden layer having 1000 nodes, third layer with 120 nodes and finally 84 nodes), and one output layer with two output nodes to classify each of the two classes (Alzheimer Atrophied and Normal Condition). For example, the input was of shape 1x260x260, the dimension of the output pattern at the first convolutional layer was 6x256x256 followed by another convolution giving output pattern as 16x252x252 which was used as an input to the series of the fully-connected layer.

2) Machine Learning Techniques along with Optimization using crow search (bio-inspired) algorithm :-
Various machine learning algorithms, were used to predict AD / CN subjects, for example Support Vector Machines (SVM), Decision Classification Tree, Random Forest Classification, K- Nearest Neighbors. 
Grid Search CV was essentially used for hyper-parameter tuning and parameter selection. The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter.

# Research Paper for reference:
https://www.inderscience.com/offers.php?id=117272
