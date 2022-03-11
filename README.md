# Fully-Convolutional-Networks-for-Semantic-Segmentation-tensorflow-2.x

This code repo is based on the paper **[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)**.  It is a fully convolutional neural network without any dense layer. The architecture is shown in figure 1 below, which includes two parts, convolution to extract features and conduct classification, and deconvolution (aka, learnable convolution transpose) to scale up the classification back to the image domain at its input size. 

![Screenshot from 2021-01-30 20-04-04](https://user-images.githubusercontent.com/58440102/106374465-925fc380-6338-11eb-8c64-9ac93bc38601.png)The input data is street views with fine annotations as labels. There are 12 classes in total. ![Screenshot from 2021-01-30 20-24-08](https://user-images.githubusercontent.com/58440102/106374558-35184200-6339-11eb-9a3d-a149172c6f84.png)

By nature, semantic segmentation is a pixel-wise classification. Instead of classifying the pixels in the input image into 0 to 11 as a single 2D output image, it outputs a (224,224, 12) probability. Each layer contains the probabilities of each pixel belonging to a class. 
I carried out both FCN8s and FCN2s in the code. Ideally, I  expect better performance with FCN2s. But the results seem to be quite similar. 




<img src="https://user-images.githubusercontent.com/58440102/106374671-31d18600-633a-11eb-8c9b-751e863abf48.png" width="400" height="350"><img src="https://user-images.githubusercontent.com/58440102/106374673-3433e000-633a-11eb-8ba4-18bbfd305b5a.png" width="400" height="350">

After training for 45 epochs and 8 classes being used, here is predicted images:![Screenshot from 2022-03-10 17-17-32](https://user-images.githubusercontent.com/58440102/157783198-9899095f-6c6f-4a50-886b-99b912b61c57.png)

In order to further improve the segmentation accuracy, more samples and a big class number would definitely help.
