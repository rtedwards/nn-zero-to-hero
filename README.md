# nn-zero-to-hero
Notes from Nerual Networks: Zero to Hero (https://www.youtube.com/watch?v=VMj-3S1tku0&amp;list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## Neural Netowrk Architecture
- Input Layer
- Hidden Layers
- Output Layer (Activation Layer?)

### Neural Network Types
- Feed Forward Networks (vanilla nerual network)
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)
- Radial Based Networks
- Generative Adversarial Networks (GAN)
- Transformers

### Hidden Layers

Different [layer types](https://en.wikipedia.org/wiki/Layer_(deep_learning)) perform different transformations on their inputs, and some layers are better suited for some tasks than others.
- Linear Layer (Linear) - regression
- Fully Connected Layer (Dense) - Multi Classification
- Convolutional Layer (CNN) - typically used for image analysis tasks. In this layer, the network detects edges, textures, and patterns. The outputs from this layer are then feed into a fully-connected layer for further processing.
- Recurrent Layer (RNN) - used for text processing with a memory function. Similar to the Convolutional layer, the output of recurrent layers are usually fed into a fully-connected layer for further processing.  
- Pooling Layer - used to reduce the size of the data input.
- Normalization Layer - adjusts the output data from previous layers to achieve a regular distribution. This results in improved scalability and model training.
- Deconvolution Layer
- Activation Layer
- Sequence Layer - A sequence input layer inputs sequence data to a neural network and applies data normalization.
- Dropout Layer - a common regularization technique to prevent overfitting.  It takes the output of the previous layer's activations and randomly sets a fraction (dropout rate) of the activations to 0, cancelling or 'dropping' them out.
   
### Output Layers (Activation Layers)
[Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
- Sigmoid
- Softmax
- Tanh
- ReLu ([Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf))
- Leaky ReLu
- Parametric ReLu ([Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852))
- ReLu6
- Binary Step
- Identity
- Swish
- Hard Swish
- Flatten

### Loss Functions (Cost Functions)
- Cross-Entropy
- Negative Log-Likelihood

## Regularizations
- L2-norm (Squared)
- L1-norm (Absolute)
- Dropout
- Batch Normalization ([Batch Normalization: Accelerating Deep Neural Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167))

## Pooling
Pooling layers are methods for reducing high dimensionality. Pooling layers provide an approach for downsampling feature maps by summarizing the presence of features in patches of the feature map.
- Max Pooling
- Average Pooling

## Concepts
- Vanishing gradients
- Dead neurons
  
## Debugging
- Vanishing gradients
- Dead neurons
