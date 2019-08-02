# Logistic Regression

Make sure to check out the [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) on coursera taught by Prof. Andrew NG, to get a better in dept understanding of Logistic Regression and Deep Learning in general. It is an absolutely brilliant course and I learnt so much from there.<br> 

Assuming now that you have understood the concept of Logistic Regression lets dive into the code.<br>
**To implement Logistic Regression you will need 5 helper function and 1 function to encapsulate the helper functions and model your algorithm.**<br>
The 6 functions are:<br>
1. sigmoid(z):<br>
   This function calculates the sigmoid value of the argument (1/0)
   
   This function has one argument : z -- A scalar or numpy array of any size
   
   It returns the sigmoid of z by using the formula<br>**s = 1/(1+e^(-z))**
2. initialize_with_zeros(dim):<br>
   This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
   This function has one argument:
    dim -- size of the w vector we want or number of parameters <br>
    
   It returns two values: <br>
    w -- initialized vector of shape (dim, 1)<br>
    b -- initialized scalar (bias)
3.  propagate(w, b, X, Y):<br>
   This function calculates the cost function and its gradient bascally implementing a Forward and Backward propagation step.<br>
   The argurments passed are:<br>
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)<br>
    b -- bias, a scalar<br>
    X -- data of size (num_px * num_px * 3, number of examples)<br>
    Y -- true "label" vector (containing 0 if negative data, 1 if positive data) of size (1, number of examples)<br>
    It returns 3 values:<br>
    cost -- negative log-likelihood cost for logistic regression<br>
    dw -- gradient of the loss with respect to w, thus same shape as w<br>
    db -- gradient of the loss with respect to b, thus same shape as b

4. optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):<br>
   This function is used to optimise and update the values of w and b ny running the gradient descent algorithm.<br>
   
   The arguments passed:<br>
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)<br>
    b -- bias, a scalar<br>
    X -- data of shape (num_px * num_px * 3, number of examples)<br>
    Y -- true "label" vector (containing 0 if negative data, 1 if positive data), of shape (1, number of examples)<br>
    num_iterations -- number of iterations of the optimization loop<br>
    learning_rate -- learning rate of the gradient descent update rule<br>
    print_cost -- True to print the loss every 100 steps
   
   The function returns:<br>
    params -- dictionary containing the weights w and bias b<br>
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function<br>
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
5. predict(w, b, X):<br>
   This function is used to predict whether the label is 0 or 1 using logistic regression parameters (w, b) that is learned.<br>
   
   The arguments passed:<br>
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)<br>
    b -- bias, a scalar<br>
    X -- data of size (num_px * num_px * 3, number of examples)<br>
   
   The values returned:<br>
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
6. model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):<br>
   This function brings together all the building blocks of Logistic Regression and places them all together to form a model.<br>
   The arguments passed:<br>
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)<br>
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)<br>
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)<br>
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)<br>
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters<br>
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()<br>
    print_cost -- Set to true to print the cost every 100 iterations<br>
   The function returns:<br>
   d -- dictionary containing information about the model.


