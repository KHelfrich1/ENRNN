'''
Eigenvalue Normalization RNN (ENRNN) architecture on the copying
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

'''
Input is a sequence of digits to copy, followed by a string of T 0s, 
a marker (we use 9) to begin printing the copied sequence, and more zeroes.
Target output is all zeroes until the marker, at which point the machine
should print the sequence it saw at the beginning.

Example for T = 10 and n_sequence = 5:

Input
3 6 5 7 2 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0
Target output
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 6 5 7 2
'''


# Network parameters
n_hidden_scoRNN = 172
n_neg_ones = 52
n_hidden_ENRNN = 20 
activation_function = 'modrelu'


# Sequence parameters
T = 2000                # Number of zeros to put between sequence and marker
n_sequence = 10         # Length of sequence to copy
n_input = 10            # Number of possible inputs (0-9)
n_classes = 9           # Number of possible output classes (0-8)
train_size = 20000
test_size = 1000
batch_size = 20
display_step = 50
training_iterations = 4000


# Optimizer Settings
optimizer = 'rmsprop'
lr = 1e-3
A_optimizer = 'rmsprop'
A_lr = 1e-5


# COMMAND LINE ARGS: SCORNN_HIDDENSIZE NEG_ONES ENRNN_HIDDENSIZE OPT LR AOPT ALR FUNC STEPS ITERS
try:
    n_hidden_scoRNN = int(sys.argv[1])
    n_neg_ones = int(sys.argv[2])
    n_hidden_ENRNN = int(sys.argv[3])
    optimizer = sys.argv[4]
    lr = float(sys.argv[5])
    A_optimizer = sys.argv[6]
    A_lr = float(sys.argv[7])
    activation_function = sys.argv[8]
    T = int(sys.argv[9])
    training_iterations = int(sys.argv[10])
except IndexError:
    pass


# Name of save string
savestring = 'ENRNN_scoRNN_{:d}_{:d}_ENRNN_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_T_{:d}_iters_{:d}_copying'.format(n_hidden_scoRNN, \
                 n_neg_ones, n_hidden_ENRNN, optimizer, lr, A_optimizer, A_lr, T, training_iterations)
print('\n')
print(savestring)
print('\n')


# Computing the number of steps and baseline
n_steps = 2*n_sequence + T
baseline = n_sequence*np.log(n_classes-1)/(T + 2*n_sequence)


# Creating scaling matrix D
D1 = np.diag(np.concatenate([np.ones(n_hidden_scoRNN - n_neg_ones), \
             -np.ones(n_neg_ones)]))


# Check to see if "experiments" folder exists, if not it create it. 
if not os.path.exists('./experiments/'):
    os.makedirs('./experiments/')


# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Creating copying problem data without zeros.   
training_seq = np.random.randint(1, high=9, size=(train_size, n_sequence))
test_seq = np.random.randint(1, high=9, size=(test_size, n_sequence))
test_seq = np.split(test_seq, 10)


def copying_data(T, seq):
    n_data, n_sequence = seq.shape
        
    zeros1 = np.zeros((n_data, T-1))
    zeros2 = np.zeros((n_data, T))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))
    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
    
    return x, y


# Defining RNN architecture
def RNN(x):
    
    rnn_cell = ENRNNCell(n_hidden_scoRNN, D1, n_hidden_ENRNN, activation_function, 'coupled')
    
    # Place RNN cell into RNN
    rnnoutput, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    
    # Last layer, linear
    with tf.variable_scope("output"):
        weights = tf.get_variable("weights", shape=[n_hidden_scoRNN + n_hidden_ENRNN, n_classes])
        biases = tf.get_variable("bias", shape=[n_classes])
    
    temp_out = tf.map_fn(lambda r: tf.matmul(r, weights), rnnoutput)
    output_data = tf.nn.bias_add(temp_out, biases)
    
    return output_data
    

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
    if sigma > 1.0 or enrnn_begin == 1:
        M_bar = M/sigma       
        enrnn_begin = 1
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


## Plotting loss & accuracy
#def graphlosses(xax, tr_loss, te_loss, baseline):      

#    plt.gca().set_title('Loss')
#    plt.plot(xax, tr_loss, label='training loss')
#    plt.plot(xax, te_loss, label='test loss')
#    plt.plot(scoRNN_results[:,0], scoRNN_results[:,1], label='scoRNN')
#    plt.plot(full_results[:,0], full_results[:,1], label='Full-Capacity')
#    plt.plot((xax[0], xax[-1]), (baseline, baseline), linestyle='-')
#    plt.ylim([0,baseline*2])
#    plt.legend(loc='upper right', prop={'size':6})
#    plt.savefig('./experiments/' + savestring + '.png')
#    plt.clf()
# 
#    return


# Graph input
x = tf.placeholder("int32", [None, n_steps])
y = tf.placeholder("int64", [None, n_steps])
input_data = tf.one_hot(x, n_input, dtype=tf.float32)    


# Assigning to RNN function
pred = RNN(input_data) 

    
# Define loss object   
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))


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
    iteration = 1
    while iteration <= training_iterations:
        
        # Gathering input data
        batch_seq = training_seq[((iteration-1)*batch_size) % train_size:(((iteration-1)*batch_size) % train_size) + batch_size]
        batch_x, batch_y = copying_data(T, batch_seq)

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


        # Evaluating model
        if iteration % display_step == 0: 

            # Evaluating train loss of model
            train_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            
            # Evaluating test loss of model
            test_metrics = []
            for seq in test_seq:
                tseq_x, tseq_y = copying_data(T, seq)
                test_metrics.append(sess.run(cost, feed_dict={x: tseq_x, y: tseq_y}))
                test_loss = np.mean(test_metrics, axis=0)
        

            # Printing results
            print('\n')
            print("Completed Iteration: ", iteration)
            print("Training Loss: ", train_loss)
            print("Test Loss: ", test_loss)
            print("Baseline: ", baseline)            
            print('\n')
      
       
            # Saving data
            train_loss_plt.append(train_loss)
            test_loss_plt.append(test_loss)
            np.savetxt('./experiments/' + savestring + '_train_loss.csv', \
                       train_loss_plt, delimiter = ',')
            np.savetxt('./experiments/' + savestring + '_test_loss.csv', \
                       test_loss_plt, delimiter = ',')
                   
        iteration += 1
    
    print("Optimization Finished!")        
