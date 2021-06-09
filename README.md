# Camera autofocus methods and lens evaluation 相机调焦算法，图像清晰度评估算子
## Introduction
本项目提供37种不同的图像卷积算子，用来评估图像的锐度、清晰度，并给出了比较不同算子和算法的方法，该方法可高效、准确的找到合适的图像清晰度评估算法，来适用于不同应用、不同类型和内容的图片，并根据最佳清晰度确定相机镜头的调焦角度。
This project provides a fast tool as well as the source code to find the best metric to adjust the camera focus that can be used in your camera based application. Not only does it decribe the advantages of each method and their comparison in evaluation the lens of your products' camera hardware, but also the sample results, validation patterns, images of the patterns under continous changing lens rotation angles are provided and visualized. The evaluation contains characterization of the lenses with respect to Focus Of View, image quality, and deformations caused due to misaligned lenses or lens characteristics. Meanwhile, different camera autofocus algorithms are compared and assessed for the suitability of the specific patterns and purposes. 

# 原理解释的参考文件
图像清晰度的评价算法的原理可参考下文链接https://blog.csdn.net/zqx951102/article/details/82790000
## Project Structure
The project folder is consisted of two folders, which are /scripts and /Test_results. Inside of /scripts, the functions and contents of the scripts are described as follows:
* **iBin_Autofocus_Metrics.py** 37 different types of metrics, including the reknowed kernels such as 
* "Variance of Median", 均值方差
* "Tenengrad Scharr", Tenengrad梯度方法，Scharr、Sobel均为图像特征提取算子。
* "Tenengrad Sobel", 同上
* "Laplacian", 拉普拉斯算子，梯度算法，相比sobel、scharr 算子，权重更加两极化
* **Read_metric_from_Excel.py** a script that loads the scores after evaluation each metric on the given samples images, and draws the curves which indicate the ideal lens rotation angle, in another way, the best focus angle. 
运用此脚本程序，每个校验pattern图片将得到一个得分，
Inside of /patterns the sample images of circle patterns can be seen. For example, see below:
![circlesPattern001](https://user-images.githubusercontent.com/60941643/120919650-4610ff80-c6ed-11eb-90dd-a4033fef6ce4.PNG)

## 图片集，样张集
该图片集采集于不同调焦角度下的校验样本图像，可参考其数量、调焦跨度来为相机或者光学传感器进行镜头调焦程序的设计。
## Prerequisites
The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Python 3
Python 3 is used when this project is constructed. This project requires the following packages to be installed: 
Python 3 can be acquired at the [official website](https://www.python.org/), download page. 
 To check which packages have already been installed in your python 3 environment, you can run the following command in your
 shell or powershell terminal:
 ```
python3 freeze
```
### Verification of preparing the dependencies


## Running the scripts

## Author

* **Yao Zhang** - *Initial work* - [yancy-zh](https://github.com/yancy-zh)


