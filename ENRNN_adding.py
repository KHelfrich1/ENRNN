'''
Eigenvalue Normalization RNN (ENRNN) architecture on the adding
problem using TensorFlow library.   
'''


# Import modules
from __future__ import print_function
from scipy.linalg import schur
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import sys
import os

from ENRNN import *


# Network parameters
n_hidden_scoRNN = 96
n_neg_ones = 29
n_hidden_ENRNN = 64 
activation_function = 'modrelu'


# Sequence parameters
n_steps = 750            # Total sequence length
n_input = 2              # Number of possible inputs (2 sequences)
n_classes = 1            # One output (sum of two numbers)
train_size = 100000
test_size = 10000
batch_size = 50
display_step = 100
training_epochs = 6

# Optimizer Settings
optimizer = 'rmsprop'
lr = 1e-4
A_optimizer = 'rmsprop'
A_lr = 1e-4

# COMMAND LINE ARGS: SCORNN_HIDDENSIZE NEG_ONES ENRNN_HIDDENSIZE OPT LR AOPT ALR FUNC STEPS EPOCHS
try:
    n_hidden_scoRNN = int(sys.argv[1])
    n_neg_ones = int(sys.argv[2])
    n_hidden_ENRNN = int(sys.argv[3]) 
    optimizer = sys.argv[4]
    lr = float(sys.argv[5])
    A_optimizer = sys.argv[6]
    A_lr = float(sys.argv[7])
    activation_function = sys.argv[8]
    n_steps = int(sys.argv[9])
    training_epochs = int(sys.argv[10])
except IndexError:
    pass


# Name of save string
savestring = 'ENRNN_scoRNN_{:d}_{:d}_ENRNN_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_seq_{:d}_epochs_{:d}'.format(n_hidden_scoRNN, \
n_neg_ones, n_hidden_ENRNN, optimizer, lr, A_optimizer, A_lr, n_steps, training_epochs)
print('\n')
print(savestring)
print('\n')


# Creating scaling matrix D1
D1 = np.diag(np.concatenate([np.ones(n_hidden_scoRNN - n_neg_ones), \
             -np.ones(n_neg_ones)]))


# Check to see if "experiments" folder exists, if not it creates it. 
if not os.path.exists('./experiments/'):
    os.makedirs('./experiments/')


# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Generates Synthetic Data
def Generate_Data(size, length):
    
    # Random sequence of numbers
    x_random = np.random.uniform(0,1, size = [size, length])

    # Random sequence of zeros and ones
    x_placeholders = np.zeros((size, length))
    firsthalf = int(np.floor((length-1)/2.0))
    for i in range(0,size):
        x_placeholders[i, np.random.randint(0, firsthalf)] = 1
        x_placeholders[i, np.random.randint(firsthalf, length)] = 1

    # Create labels
    y_labels = np.reshape(np.sum(x_random*x_placeholders, axis=1), (size,1))
    
    # Creating data with dimensions (batch size, n_steps, n_input)
    data = np.dstack((x_random, x_placeholders))
    
    return data, y_labels



# Defining RNN architecture
def RNN(x):

    # Create RNN cell
    rnn_cell = ENRNNCell(n_hidden_scoRNN, D1, n_hidden_ENRNN, activation_function, 'coupled')
    
    # Place RNN cell into RNN
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    rnnoutput = outputs[:,-1]
        
    # Last layer, linear
    output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None)
    

    return output
    

# Used to calculate Cayley Transform derivative
def Cayley_Transform_Deriv(grads, A, W):
    
    # Calculate update matrix
    I = np.identity(grads.shape[0])
    Update = np.linalg.lstsq((I + A).T, np.dot(grads, D1 + W.T), rcond=None)[0]
    DFA = Update.T - Update
    
    return DFA

# Used to calculate derivative of spectral normalized matrix
def EN_Deriv(M_bar, M_bar_grads, sigma, u1, v1, alpha, beta):
    
    # Calculate update matrix
    ones_left = np.ones([1, M_bar_grads.shape[0]])
    ones_right = np.ones([M_bar_grads.shape[1], 1])
    update_matrix = np.matmul(np.conj(v1), u1.T)/np.matmul(np.conj(v1).T, u1)
   
    Update = 1.0/sigma*(M_bar_grads - 1.0/sigma*np.matmul(np.matmul(ones_left, M_bar_grads*M_bar), ones_right)*(alpha*np.real(update_matrix) + beta*np.imag(update_matrix)))
    
    return Update


# Used to normalize the recurrent weight matrix
def eigenvalue_normalize(M, enrnn_begin=None):

    # Computing eigenvalues and eigenvectors of M
    T, Z = schur(M, output='complex') 
    eigenvalues = np.diagonal(T, offset=0)
    modulus_eigenvalues = np.absolute(eigenvalues)

    # Computing modulus of eigenvalues to find largest (modulus)
    largest_eigen_arg = np.argmax(modulus_eigenvalues)
    largest_eigen = eigenvalues[largest_eigen_arg]
    alpha = np.real(largest_eigen)
    beta = np.imag(largest_eigen)
    sigma = modulus_eigenvalues[largest_eigen_arg]
    
    # Computing right/left eigenvectors
    if largest_eigen_arg > 1:
        T11 = T[:largest_eigen_arg, :largest_eigen_arg]
        T12 = T[:largest_eigen_arg, largest_eigen_arg]
        T33 = T[largest_eigen_arg+1:, largest_eigen_arg+1:]
        T23 = T[largest_eigen_arg, largest_eigen_arg+1:]

        I = np.eye(T11.shape[0])

        x1_top = -np.linalg.lstsq(T11 - largest_eigen*I, T12, rcond=None)[0]
        x1_top = x1_top.reshape([-1, 1])
        x1_bottom = np.concatenate([np.ones([1,1]), np.zeros((n_hidden_ENRNN-1-x1_top.shape[0], 1))], axis=0)
        x1 = np.concatenate([x1_top, x1_bottom], axis=0)
        u1 = np.matmul(Z, x1)

        # Computing left eigenvector
        I = np.eye(T33.shape[0])

        x1_bottom = -np.linalg.lstsq(np.conj(np.transpose(T33-largest_eigen*I)), np.conj(T23.T), rcond=None)[0]
        x1_bottom = x1_bottom.reshape([-1,1])
        x1_top = np.concatenate([np.zeros((n_hidden_ENRNN-1-x1_bottom.shape[0],1)), np.ones((1,1))], axis=0)
        x1 = np.concatenate([x1_top, x1_bottom], axis=0)
        v1 = np.matmul(Z, x1)


    elif largest_eigen_arg > 0:
        T11 = T[0,0]
        T12 = T[0,1]
        x1_top = np.array([-T12/(T11-largest_eigen), 1])
        x1 = np.concatenate([x1_top, np.zeros(n_hidden_ENRNN-2)], axis=0)
        x1 = x1.reshape([-1, 1])
        u1 = np.matmul(Z, x1)

        T23 = T[1,2:]
        T33 = T[2:,2:]
        I = np.eye(n_hidden_ENRNN-2)
        x1_bottom = -np.linalg.lstsq(np.conj(np.transpose(T33 - largest_eigen*I)),np.conj(T23.T), rcond=None)[0]
        x1_bottom = x1_bottom.reshape([-1,1])
        x1 = np.concatenate([ np.zeros((1,1)), np.ones((1,1)), x1_bottom], axis=0)
        v1 = np.matmul(Z, x1)   
    
    else:
        u1 = Z[:,0] 
        u1 = u1.reshape([-1, 1])       

        T22 = T[1:,1:]
        T12 = T[0,1:]
        I = np.eye(n_hidden_ENRNN-1)
        x1_bottom = -np.linalg.lstsq(np.conj(np.transpose(T22-largest_eigen*I)), np.conj(T12.T), rcond=None)[0]
        x1_bottom = x1_bottom.reshape([-1, 1])
        x1 = np.concatenate([np.ones((1,1)), x1_bottom], axis=0)
        v1 = np.matmul(Z, x1)

    # Computing M_bar
    if sigma > 1.0 or enrnn_begin == 1.0:
        M_bar = M/sigma
        enrnn_begin = 1.0
    else:
        M_bar = np.matrix.copy(M)
        enrnn_begin = 0

    return M_bar, sigma, u1, v1, alpha, beta, enrnn_begin



# Used to make the hidden to hidden weight matrix
def makeW(A):
    # Computing hidden to hidden matrix using the relation 
    # W = (I + A)^-1*(I - A)D
    
    I = np.identity(A.shape[0])
    W = np.dot(np.linalg.lstsq(I+A, I-A, rcond=None)[0],D1)  

    return W


# Generating training and test data
x_train, y_train = Generate_Data(train_size, n_steps)
test_data, test_label = Generate_Data(test_size, n_steps)


# Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Assigning to RNN function
pred = RNN(x) 

    
# Define loss object   
cost = tf.reduce_mean(tf.squared_difference(pred, y))


# Optimizers/Gradients
optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                  'adagrad' : tf.train.AdagradOptimizer,
                  'rmsprop' : tf.train.RMSPropOptimizer,
                  'sgd' : tf.train.GradientDescentOptimizer}

# A optimizer
opt1 = optimizer_dict[A_optimizer](learning_rate=A_lr)

# All other weights optimizer
opt2 = optimizer_dict[optimizer](learning_rate=lr)

# Training operations
W1_var = [v for v in tf.trainable_variables() if 'W1:0' in v.name][0]
A1_var = [v for v in tf.trainable_variables() if 'A1:0' in v.name][0]    
W2_bar_var = [v for v in tf.trainable_variables() if 'W2_bar:0' in v.name][0]
W2_var = [v for v in tf.trainable_variables() if 'W2:0' in v.name][0]
othervarlist = [v for v in tf.trainable_variables() if v not in [W1_var, A1_var, W2_bar_var, W2_var]]
   
# Getting gradients
grads = tf.gradients(cost, othervarlist + [W1_var] + [W2_bar_var])
  
# Applying gradients to non-recurrent weights
with tf.control_dependencies(grads):
    applygrad1 = opt2.apply_gradients(zip(grads[:len(othervarlist)], \
                 othervarlist))  
    
# Updating variables
new_W1 = tf.placeholder(tf.float32, W1_var.get_shape())
update_W1 = tf.assign(W1_var, new_W1)

new_W2_bar = tf.placeholder(tf.float32, W2_bar_var.get_shape())
update_W2_bar = tf.assign(W2_bar_var, new_W2_bar)
    
# Applying recurrent gradients
grad_A1 = tf.placeholder(tf.float32, A1_var.get_shape())
applygrad_A1 = opt1.apply_gradients([(grad_A1, A1_var)])
    
grad_W2 = tf.placeholder(tf.float32, W2_var.get_shape())
applygrad_W2 = opt2.apply_gradients([(grad_W2, W2_var)])

# Initializing the variables
init = tf.global_variables_initializer()


# Plotting lists
train_loss_plt = []
test_loss_plt = []


# Training
with tf.Session() as sess:
    
    # Initializing the variables
    sess.run(init)
                
    # Get initial W1, A1, W2, W2_bar
    W1, A1, W2, W2_bar = sess.run([W1_var, A1_var, W2_var, W2_bar_var])
    W2_bar = np.matrix.copy(W2)
    sigma = 0
    u1 = 0
    v1 = 0
    alpha = 0
    beta = 0     
    enrnn_begin = 0
    sess.run(update_W2_bar, feed_dict={new_W2_bar : W2_bar})
    
    # Keep training until reach number of iterations
    epoch = 1
    while epoch <= training_epochs:
        
        # Shuffling Data
        randomize = np.arange(x_train.shape[0])
        np.random.shuffle(randomize)
        x_train = x_train[randomize, :]
        y_train = y_train[randomize] 

        # Keep training until reach max iterations
        step = 1        
        while step * batch_size <= train_size:            
            
            # Getting input data
            batch_x = x_train[(step-1)*batch_size:step*batch_size,:,:]
            batch_y = y_train[(step-1)*batch_size:step*batch_size]   

            W1_grads, W2_bar_grads = sess.run([grads[-2], grads[-1]], feed_dict = {x: batch_x, y: batch_y})
            _ = sess.run(applygrad1, feed_dict = {x: batch_x, y: batch_y})
                
            DFA1 = Cayley_Transform_Deriv(W1_grads, A1, W1)
            if enrnn_begin == 1.0:
                DFW2 = EN_Deriv(W2_bar, W2_bar_grads, sigma, u1, v1, alpha, beta)          
            else:
                DFW2 = np.matrix.copy(W2_bar_grads)

            sess.run(applygrad_A1, feed_dict = {grad_A1: DFA1})
            sess.run(applygrad_W2, feed_dict = {grad_W2: DFW2})

            A1 = sess.run(A1_var)
            W2 = sess.run(W2_var)

            W1 = makeW(A1)
            W2_bar, sigma, u1, v1, alpha, beta, enrnn_begin = eigenvalue_normalize(W2, enrnn_begin)

            sess.run(update_W1, feed_dict={new_W1: W1})
            sess.run(update_W2_bar, feed_dict = {new_W2_bar: W2_bar})

        
            # Evaluating the MSE of the model
            if step % display_step == 0:
                
                # Evaluating train and test MSE.   
                train_mse = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                test_mse = sess.run(cost, feed_dict={x: test_data, y: test_label})
                

                # Printing results
                print('\n')
                print("Epoch:", epoch)
                print("Percent complete:", step*batch_size/train_size) 
                print("Training Minibatch MSE:", train_mse)
                print("Test MSE:", test_mse)
                    

                # Saving data
                train_loss_plt.append(train_mse)
                test_loss_plt.append(test_mse)
                    


                np.savetxt('./experiments/' + savestring + '_train_MSE.csv', train_loss_plt, \
                		  delimiter=',')
                np.savetxt('./experiments/' + savestring + '_test_MSE.csv', test_loss_plt, \
		                 delimiter = ',')
             
            step += 1                   
        epoch += 1
    
    print("Optimization Finished!")        
