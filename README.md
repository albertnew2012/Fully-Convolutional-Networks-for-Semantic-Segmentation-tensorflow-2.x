# Fully-Convolutional-Networks-for-Semantic-Segmentation-tensorflow-2.x

This code repo is based on the paper **[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)**.  It is a fully convolutional neural network without any dense layer. The architecture is shown in figure 1 below, which includes two parts, convolution to extract features and conduct classification, and deconvolution (aka, learnable convolution transpose) to scale up the classification back to the image domain at its input size. 

![Screenshot from 2021-01-30 20-04-04](https://user-images.githubusercontent.com/58440102/106374465-925fc380-6338-11eb-8c64-9ac93bc38601.png)The input data is street views with fine annotations as labels. There are 12 classes in total. ![Screenshot from 2021-01-30 20-24-08](https://user-images.githubusercontent.com/58440102/106374558-35184200-6339-11eb-9a3d-a149172c6f84.png)

By nature, semantic segmentation is a pixel-wise classification. Instead of classifying the pixels in the input image into 0 to 11 as a single 2D output image, it outputs a (224,224, 12) probability. Each layer contains the probabilities of each pixel belonging to a class. 
I carried out both FCN8s and FCN2s in the code. Ideally, I  expect better performance with FCN2s. But the results seem to be quite similar. 




<img src="https://user-images.githubusercontent.com/58440102/106374671-31d18600-633a-11eb-8c9b-751e863abf48.png" width="500" height="400"><img src="https://user-images.githubusercontent.com/58440102/106374673-3433e000-633a-11eb-8ba4-18bbfd305b5a.png" width="500" height="400">

After training for 45 epochs, here is predicted images:![Screenshot from 2021-01-30 20-48-37](https://user-images.githubusercontent.com/58440102/106374953-9a216700-633c-11eb-942b-72480282d7d3.png)

Sorry, some classes, like the pedestrians and cyclists are poorly segmented from the input images. I think it might be due to the insufficient representations in the training data, it may also be caused by its relatively small size in the image. It can be further improved by adding more samples of pedestrians and cyclists and refine the architecture.
