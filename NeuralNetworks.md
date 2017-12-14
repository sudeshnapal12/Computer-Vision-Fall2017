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
    
    * Regularization loss because W, 2*W, 100*W can use anything. Control weights.
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
     * SVM doesnot care if loss score difference is within delta. Softmax optimizes loss. Softmax micromanages.
     * eg. Car classifier => classifies cars from humans faces etc. SVM won't optimize loss for car and frog.
   
