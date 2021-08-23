# DeepNCI

## Introduction

 DeepNCI is a tool for predicting Non-covalent interaction(NCI). In this tool, I mainly use deep-learning architecture including 3D-CNN and DNN. Because of the strong ability of detecting can capturing the characteristic related to NCI in some extent of 3D-CNN, we build it for the feature of electro density of moleculars. And I use DNN to detect information of chemical properties.

**tf_Conv3D_elect.py** is a single 3D-CNN network specialised for electro density. 

**tf_DNN_chem.py** is a DNN network for chemical properties. 

**DeepNCI.py** is used to build the final network, combines above two networks and achieves the best performance.

**tf_Conv3D_transfer.py** is a transform network for chemical databases using DeepNCI model.
 
## Build environment 

Python: 3.6

Tensorflow: 1.12.0



