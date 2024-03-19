#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                m.data[i][j] = 1. / (1 + exp(-x));
            } else if (a == RELU){
                m.data[i][j] = x > 0 ? x : 0;
            } else if (a == LRELU){
                m.data[i][j] = x > 0 ? x : 0.1*x;
            } else if (a == SOFTMAX){
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            for(j = 0; j < m.cols; ++j){
                m.data[i][j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                d.data[i][j] *= x * (1 - x);
            } else if (a == RELU){
                d.data[i][j] *= x > 0 ? 1 : 0;
            } else if (a == LRELU){
                d.data[i][j] *= x > 0 ? 1 : 0.1;
            } else if (a == SOFTMAX){
                d.data[i][j] *= 1;
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  


    matrix out = matrix_mult_matrix(in, l->w);
    activate_matrix(out, l->activation);


    free_matrix(l->out);
    l->out = out;       
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    gradient_matrix(l->out, l->activation, delta);


    // 1.4.2
    free_matrix(l->dw); 
    matrix transposed_in = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(transposed_in, delta);
    l->dw = dw;
    free_matrix(transposed_in); 


    
    // 1.4.3
    matrix transposed_w = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, transposed_w);
    free_matrix(transposed_w);

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // Assuming l->dw (gradient), l->w (current weights), and l->v (previous update) are initialized.

    // Calculate weight decay term: λw_t
    matrix weight_decay_term = copy_matrix(l->w);
    for (int i = 0; i < weight_decay_term.rows; ++i) {
        for (int j = 0; j < weight_decay_term.cols; ++j) {
            weight_decay_term.data[i][j] *= decay;
        }
    }

    // Calculate momentum term: mΔw_{t-1}
    matrix momentum_term = copy_matrix(l->v);
    for (int i = 0; i < momentum_term.rows; ++i) {
        for (int j = 0; j < momentum_term.cols; ++j) {
            momentum_term.data[i][j] *= momentum;
        }
    }

    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    matrix dw_t = copy_matrix(l->dw);
    for (int i = 0; i < dw_t.rows; ++i) {
        for (int j = 0; j < dw_t.cols; ++j) {
            dw_t.data[i][j] -= weight_decay_term.data[i][j];
            dw_t.data[i][j] += momentum_term.data[i][j];
        }
    }

    // Update l->w: w_t = w_t + rate * Δw_t
    for (int i = 0; i < l->w.rows; ++i) {
        for (int j = 0; j < l->w.cols; ++j) {
            l->w.data[i][j] += rate * dw_t.data[i][j];
        }
    }

    // Save Δw_t to l->v for the next iteration
    free_matrix(l->v);
    l->v = copy_matrix(dw_t);

    // Free any intermediate results to avoid memory leaks
    free_matrix(weight_decay_term);
    free_matrix(momentum_term);
    free_matrix(dw_t);
}


// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
// 2.1.1 What are the training and test accuracy values you get? Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// Training : 90.34 %; Test: 90.91 %; We are interested in both training and testing accuracy because we want to know how well our model is performing on the training data and how well it generalizes to unseen data. The training accuracy tells us how well our model is fitting the training data, while the testing accuracy tells us how well our model is generalizing to unseen data. If the training accuracy is high but the testing accuracy is low, it means that our model is overfitting to the training data and not generalizing well to unseen data. If the training accuracy is low, it means that our model is not fitting the training data well and we need to modify the model.

// 2.1.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// When the learning rate is too high, the loss during training oscillates and does not converge. When the learning rate is too low, the loss during training decreases very slowly and does not converge. The best learning rate is 10^-1, which converges quickly and has the lowest loss during training. The final model accuracy is highest (Test: 91.71 %; Training: 92.07 %) when the learning rate is 10^-1. 

// 2.1.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// The weight decay parameter slightly affects the final model training and test accuracy. The best weight decay parameter is 10^-4, which has the highest test accuracy (Test: 91.71 %; Training: 92.07 %). Theoretically, weight decay helps to prevent overfitting by penalizing large weights. It helps to reduce the variance in the model and improve the generalization to unseen data.


// 2.2.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// In this experiment, the learning rate and weight decay were set to 0.1 and 0.0001 respectively.
// Logistic: Test: 94.45%, Training: 94.49%.
// Relu: Test: 94.33%, Training: 94.92%.
// LRELU: Test: 94.56%, Training: 94.99%.

// The best activation function is the LRELU activation function, which has the highest test accuracy. 

// 2.2.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// In this experiment, I use LRELU as the activation function and set the decay to 0.
// The best learning rate is 10^-1, which has the highest test accuracy (Test: 94.58 %; Training: 95.13 %).

// 2.2.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// The best weight decay parameter is 10^-3, which has the highest test accuracy (Test: 94.90 %; Training: 95.27 %). Weight decay helps to prevent overfitting by penalizing large weights. It helps to reduce the variance in the model and improve the generalization to unseen data.

// 2.2.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// More layers and more iterations allow the model to learn more complex patterns in the data. The best weight decay parameter is 10^-3,  which has the highest test accuracy (Test: 97.11 %; Training: 98.31 %). 

// 3.1.1 What is the best training accuracy and testing accuracy? Summarize all the hyperparameter combinations you tried.
// In this experiment, I use 3-layer model (inputs -> 64 -> 32 -> outputs) with 3000 iterations and LRELU as the activation function. Hyperparameter combinations and results:



// lr=0.1, decay=0, Train Acc: 40.30%, Test Acc: 38.62%
// lr=0.1, decay=0.01, Train Acc: 36.90%, Test Acc: 35.61%
// lr=0.1, decay=0.001, Train Acc: 40.56%, Test Acc: 39.30%
// lr=0.1, decay=0.0001, Train Acc: 40.61%, Test Acc: 39.14%

// lr=0.01, decay=0, Train Acc: 46.29%, Test Acc: 44.77%
// lr=0.01, decay=0.01, Train Acc: 47.24%, Test Acc: 45.56%
// lr=0.01, decay=0.001, Train Acc: 47.08%, Test Acc: 45.21%
// lr=0.01, decay=0.0001, Train Acc: 46.88%, Test Acc: 44.63%


// lr=0.001, decay=0, Train Acc: 39.88%, Test Acc: 39.67%
// lr=0.001, decay=0.01, Train Acc: 39.94%, Test Acc: 39.73%
// lr=0.001, decay=0.001, Train Acc: 39.86%, Test Acc: 39.77%
// lr=0.001, decay=0.0001, Train Acc: 39.81%, Test Acc: 39.79%

// lr=0.0001, decay=0, Train Acc: 28.14%, Test Acc: 28.40%
// lr=0.0001, decay=0.01, Train Acc: 26.95%, Test Acc: 26.31%
// lr=0.0001, decay=0.001, Train Acc: 26.94%, Test Acc: 26.30%
// lr=0.0001, decay=0.0001, Train Acc: 28.12%, Test Acc: 28.37%

// Best Test Accuracy: 45.56% with parameters {'rate': 0.01, 'decay': 0.01}



