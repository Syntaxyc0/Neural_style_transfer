# Neural_style_transfer

This python project is the implementation of a Neural network to transfer the style of an image to another one, for this we use a pre-trained convolutional neural network called VGG19

## How does it work ?

1. We preprocess content and style images to convert them into tensors

2. We load our pre-trained model and select some specific features of both images

3. We iteratively apply changes to our content image and compute our proper loss which takes care of keeping the result close to the base image with the style of the second one and the global coherence of the result
