## CNNs and Transfer Learning

Your goal is to:
Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9
* Architecture (suggested, you may change it):
  * "conv1": conv2d 3x3x4, stride=1, ReLU, padding = "SAME"
  * "conv2": conv2d 3x3x8, stride=2, ReLU, padding = "SAME"
  * "pool": pool 2x2
  * "fc1": fc 16
  * "fc2": fc 10
  * "softmax": xentropy loss, fc-logits = 4 (we have 4 classes...)
* (I suggested scope "names" here so it's easier to reference)
* Optimizer: ADAM
* 5 epochs, 10 batch size

* Use your trained model’s weights on the lower 4 layers to train a classifier for the rest of MNIST (excluding 1,4,5 and 9)
  * Create new layers for the top (5 and 6)
  * Try to run as few epochs as possible to get a good classification (> 99% on test)
  * Try a session with freezing the lower layers weights, and also a session of just fine-tuning the weights.
    * Use (for speed) a constraint on the optimizer for freezing:
    ```
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc2_023678|softmax_023678")
    training_op = optimizer.minimize(loss, var_list=train_vars)
    ```
    
## Results in result.pdf
Report your:
* Test loss curve on MNIST-1459
* Test loss curve on transferred MNIST-023678:
  * with fine-tuning everything
  * with frozen layers up to fc2 (and not including)
* Final execution graph (provided code)
* **Bonus 1 (5pt)**:
  * Apply dropout regularization after conv1, conv2 and fc1
* **Bonus 2 (5pt)**:
  * Visualize the filter maps (activations) for conv1, conv2 and pool 
