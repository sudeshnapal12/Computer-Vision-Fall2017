# Notes on Neural Networks, DNNs, CNNs and RNNs.
Thanks to Karapathy for his wonderful tutorial http://cs231n.github.io/ and classes taken in SBU by Roy Shilkrot. Learnt a lot.

## 1. Image classification
Input = 3-dimensional array of numbers in cat image = 248 x 400 x 3 numbers = 297,600 numbers.
Output = label cat

#### Challenges
[[ https://github.com/sudeshnapal12/Computer-Vision-Fall2017/blob/master/images/challenges.PNG ]]

#### Algorithms
* NN
* KNN : Hyperparameter = K
* ANN

<b> NOTE: </b> Finally we need to go beyond raw pixels. </br>
eg: A dog can be seen very near a frog since both happen to be on white background. Ideally we would like images of all of the 10 classes to form their own clusters, so that images of the same class are nearby to each other regardless of irrelevant characteristics and variations (such as the background). 
However, to get this property we will have to go beyond raw pixels.

## 2. Linear Classification
* f(xi, W, b) = Wxi + b
* W is +ve if wrong prediction, -ve if correct prediction.
* Image pixel squashed to 1D vector and put into xi. W has #classes = #rows and size of xi = #columns.
* Interpretation 1 : Each row of W is Template of that class. So, class score = template * image. Image class = Highest score.
* Interpretation 2 : Like doing NN. Distance(1 image in class, image) = their dot product instead of L1/L2. 
* <b>NOTE</b> A template will contain all features of a class. </br>
eg: A horse template might have 2 headed horse to account for horses in both poses.</br>
eg: A car template might be red because most cars ended up red in dataset used to calculate template. NN uses hidden layers to account for blue cars.
* Normalize pixels

* Loss
  * SVM Classifier - Hinge loss </br>
    * Score of correct class y_i greater than incorrect class score by atleast delta/margin.
    * SVM loss = data loss + regularization
    
    L = 1/N &#931; Li + &#955; R(W) </br>
    L = 1/N &#931;&#931; [max(0, f(xi, Wj) - f(xi, Wyi) + margin] +&#955; &#931;&#931; W*W </br>
    
    * Regularization loss needed because W, 2*W, 100*W can use anything. Control weights.
    * Prefer weights uniformly diffused. </br> 
    Eg prefer w=[0.25,0.25,0.25,0.25] over w=[1,0,0,0] because lower regularization loss. </br>
    => less overfitting </br>
    => more generalization </br>
    * Delta and Lambda are hyperparameters
    * Both delta and lambda tune same thing. Trade off between data loss and regularzation loss. </br>
    Delta = 1 (safe). Exact value of delta doesnt matter much. </br>
    Large value of W => large loss score and vice versa. lamda controls values of W.
    * <b> HINGE LOSS </b> = max(0,-)
    * HINGE Squared loss = max(0,-)^2
    * Gradient not differentiable eg. hinge loss at correct class, use <b> sub-gradients </b>
   
  * Softmax Classifier - Cross entropy loss </br>
    * Probabilitistic output
    * Softmax function = (e^f_yi) / (&#955; e^fj); f_yi = score of correct label </br>
        Squashes values between 0 and 1 and sums to 1. </br>
        Normalize scores for numerical stability. </br>
    * Cross entropy loss = -log(sofmax loss) </br>
        Maximum log liklihood = minimum (negative log liklihood) = Maximum a Posteriori estimation
        Minimizinf KL divergence between 2 distributions.
    * eg. small lambda/regularization param. [1,-2,0] = [e^-1, e_-2, e^0] = [2.71,0.14,1] = normlized to [0.7,0.04,0.26] </br>
      large lambda [0.5, -1, 0] = [e^.5, e^-1, e^0] = [1.65,0.37,1] = normalized to [0.55,0.12,0.33]
   
   * SVM vs. Softmax
     * Softmax gives probabilities of each class while SVM just gives score. 
     * SVM doesnot care if loss score difference is within delta. Softmax optimizes loss. Softmax micromanages.
     * eg. Car classifier => classifies cars from humans faces etc. SVM won't optimize loss for car and frog.
   
## 3. Optimization
* Gradient = vector of slopes along each dimension.
* Step-size = learning rate = step taken in gradual hill descent in SGD = Hyperparameter.
* decrease weights in step-size along the direction of gradient.
* Gradient update
  * M1 => update gradiant after running through all samples in training set.
  * Mini-batch update => update gradient after every batch. eg. every 256 images in a training set.
  * SGD => mini-batch size = 1 => update gradient after every sample. Online training / Online GD.
  
## 4. Backpropagation
 * Chain rule => need gradients in backprop
 * backprop value = (o/p gradient) * (local gradient
 * Sigmoid function = Sigma(x) = 1/(1 + e^-x) </br>
   d/dx (sigma_x) = (1-sigma_x)(sigma_x)
 * Intution of common backprop operations:
   * add => distributes outer gradient to i/p
   * max => routes gradient => gives o/p to only one i/p
   * multiply => switches i/p and multiply gradient

## 5. Neural Networks
 * 2 layer NN example => s = W_2 max(0, W_1 * x)
 * 3 layer NN example => s = W_3 max(0,  W_2 max(0, W_1 * x))
 * If x = [3072 x 1], W1 = [100 x 3072], W2 = [10 x 100] in 2NN for a 10 class classifier. NN with 2 layers
 * Activation fuction for non-linearity. Here RELU is used. RELU = max(0, -)
 * W_1, W_2 and W_3 are learned with SGD
 * Hyperparameters = Size of hidden vectors.
 * Activation functions
   * <b> SIGMOID </b> 
     * 1/ (1+ e^-x)
     * limit between -1 and 1 
     * saturate gradients / vanishing gradients => careful while initializing weights.
     * Not zero centered
   * <b> tanh </b>
     * Vanishing gradients
     * centered around zero
     * prefer over sigmoid
   * <b> RELU </b>
     * f(x) = max(0,x)
     * Thresholding at zero
     * Not expensive
     * dead neurons while training. large gradient = -ve slope => goes through RELU => update weights such that gradeints never flow again. => dead
   * <b> Leaky RELU </b>
     * add small slope eg. 0.01 for negative values.
     * f(x)= 1(x<0)(αx) + 1(x>=0)(x)
     * Results not consistent
   * <b> Maxout </b>
     * RELU + Leaky RELU
     * f(x) = max(W_1*x + b_1, W_2*x + b_2)
 * Which type of neuron to use? RELU by moitoring learning rate and dead neurons %.
 * Full connected n/w => all FC layers
 * #weights + #biases = #learnable parameters
 * Neural network with one hidden layer can represent any continuous function.
 * Then why so many hidden layers? because you don't know constraints and want to generalize.
 * More layers => more complex functions => overfitting.
 * so use less layers? No use more layers with regularization (such as dropout, higher weiht decay).
 
 ## 6. CNNs
 * Similar to NN. CNN are applied to images.
 * NN has all FC layers. CNN has 3 types: Convolutional layers, Pooling layers nadn FC layers.
 * Conv layers => filters are used.
    * each filter is convolved across spatial dimension. Eg filter to detect edge in an image. So, the name convolution.
    * preserve spatial dimension and squash depth. 
    * Depth of Conv layer = # filters.
    * 32x32x3 i/p and 5x5 filter then, weights in conv layer = 5x5x3 + 1 = 76 weight parameters.
 * Pooling layers => Max pool
    * Reduce spatial dimension (preferable only in pool and not in conv)
    * F=3, S=2 => overlapping pooling
    * F=2, S=2 => commonly used => reduced to half.
 * FC => compute class scores.
 * eg Architecture : [I/p => Conv => Relu => Pool => FC]
    * i/p = [32x32x3], conv layer with 12 filters = [32x32x12], RELU = [32x32x12], Pool=[16x16x12], FC=[1x1x10] for 10 classes.
 * Hyperparameter => 
    * filter-size/local connectivity
    * depth
    * stride
    * padding
 * Contraints 
    * Zero-padding for (i/p dimension = o/p dmension), constant size => P = (F-1)/2
    * Dimension of o/p layer = (W - F + 2P)/S + 1
 * TRICKS
    * Parameter sharing => reduce #parameters => same weights in a depth slice. eg. 55x55*96 o/p layer has 96 depth slices of 55x55. If filter of 11x11x3 is used, #parameters = (11x11x3)x96 + 96
    * Common architectures
       * I/p -> FC (linear classifier)
       * I/p -> Conv -> RELU -> FC
       * I/p -> [Conv -> RELU -> Pool]*2 -> FC -> RELU -> FC
       * I/p -> [Conv -> RELU -> Conv -> RELU -> Pool]*3 -> [FC -> RELU]*2 -> FC
       
## 7. Famous CNN architectures
  1. LeNet
  2. AlexNet
  3. GoogLeNet/Inception
  4. ResNet
  5. R-CNN
  6. YOLO
  
## 8. Overfitting prevention
* Data
  1. Data Augmentation
  2. Normalization
* Model
  1. Regularization  Constrain model parameters.
     * L2 regularization 
     * L1 regularization
     * ElasticNet regularization => L1+L2 
  2. Dropout - Reduce DOF
  3. DropConnect
  4. Early stopping
  
 ## 9. Transfer learning
 * Proxy problems
 * Solutions
   * #### Autoencoders => (Encoder & Decoder) => Reconstruct input in output
     * Encoder = Convolution & Pooling
     * Decoder = Deconvolution & Unpooling
   * #### Semantic Segmentation
   * #### Varational Autoencoders => Probabilitsic Autoencoder
   * #### GANs (Generative Adversial Networks) => (Generator & Discriminator)
     * Arithmetics => (smiling woman - neural woman + neural man = smiling man)
     * Recoloring => Shape to image

## 10. RNN (Recurrent Neural Networks)
  * Handle i/p wth time sequence (video), o/p sequence, sequence to sequence
  * Backprop through time
  * Visual Applications
    * Image captioning (one i/p => multiple o/ps) 
    * Sentiment analysis (multiple i/ps => one o/p)
    * Video captioning/machine translation (multiple i/p => multiple o/p)
  * Examples
    * char RNNs => generate sentences
    * generate c code/latex code
  * Problems
    * Big gaps/long sequences => exploding gradients/vanishing gradientts
  * Solution => <b> LSTMs </b>
  
## 11. LSTMs (Long Short Term Memory Cells) => Forget gate
  * Visual Applications
    * Image captioning
    * Image captioning with attention
    * Visual Question Answering
    * VQA wih Attention
