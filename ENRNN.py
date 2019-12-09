from tensorflow.python.ops.rnn_cell_impl import RNNCell 

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs



class ENRNNCell(RNNCell):
    """
    Eigenvalue Normalized Recurrent Neural Network (ENRNN) Cell
    """

    def __init__(self, hidden_size_scoRNN, D1, hidden_size_ENRNN, activation_function=None, coupling=None):
        
        self._hidden_size_scoRNN = hidden_size_scoRNN
        self._hidden_size_ENRNN = hidden_size_ENRNN        
        self._activation_function = activation_function
        self._D1 = D1
        self._coupling = coupling
      
        # Initialization of scoRNN component      
        
        # Initialization of skew-symmetric matrices
        s1 = np.random.uniform(0, np.pi/2.0, \
        size=int(np.floor(self._hidden_size_scoRNN/2.0)))
        s1 = np.sqrt((1.0 - np.cos(s1))/(1.0 + np.cos(s1)))
        z1 = np.zeros(s1.size)
        if self._hidden_size_scoRNN % 2 == 0:
            diag1 = np.hstack(tuple(zip(s1, z1)))[:-1]
        else:
            diag1 = np.hstack(tuple(zip(s1,z1)))
        A1_init = np.diag(diag1, k=1)
        A1_init = A1_init - A1_init.T
        A1_init = A1_init.astype(np.float32)

        self._A1 = tf.get_variable("A1", [self._hidden_size_scoRNN, self._hidden_size_scoRNN], \
                   initializer = init_ops.constant_initializer(A1_init))


        # Initialization of scoRNN hidden to hidden matrix
        I1 = np.identity(self._hidden_size_scoRNN)
        Z1_init = np.linalg.lstsq(I1 + A1_init, I1 - A1_init, rcond=None)[0].astype(np.float32)
        W1_init = np.matmul(Z1_init, self._D1)

        self._W1 = tf.get_variable("W1", [self._hidden_size_scoRNN, self._hidden_size_scoRNN], \
                                   initializer = init_ops.constant_initializer(W1_init))


        # Initialization of eigenvalue normalized reccurent matrix
        
        # Initialization of skew-symmetric matrices
        s2 = np.random.uniform(0, np.pi/2.0, \
        size=int(np.floor(self._hidden_size_ENRNN/2.0)))
        s2 = np.sqrt((1.0 - np.cos(s2))/(1.0 + np.cos(s2)))
        z2 = np.zeros(s2.size)
        if self._hidden_size_ENRNN % 2 == 0:
            diag2 = np.hstack(tuple(zip(s2, z2)))[:-1]
        else:
            diag2 = np.hstack(tuple(zip(s2,z2)))

        A2_init = np.diag(diag2, k=1)
        A2_init = A2_init - A2_init.T
        A2_init = A2_init.astype(np.float32)

        
        # Initialization of scaling matrix D2
        D2_diag = np.random.uniform(-1.0, 1.0, size=[int(np.floor(self._hidden_size_ENRNN/2.0))])
        j =  0
        while j < int(np.floor(self._hidden_size_ENRNN/2.0)):
            r = 2*j+1
            s = 2*j
            D2_diag = np.insert(D2_diag, r, D2_diag[s])
            j += 1
        if self._hidden_size_ENRNN % 2 != 0:
            rho_0 = np.random.uniform(-1, 1, size=[1,])
            D2_diag = np.insert(D2_diag, 0, rho_0)
        D2_init = np.diag(D2_diag)
              
        # Initialization of recurrent matrix
        I2 = np.identity(self._hidden_size_ENRNN)
        Z2_init = np.linalg.lstsq(I2 + A2_init, I2 - A2_init, rcond=None)[0].astype(np.float32)
        W2_init = np.matmul(Z2_init, D2_init)

        self._W2 = tf.get_variable("W2", [self._hidden_size_ENRNN, self._hidden_size_ENRNN], \
                  initializer = init_ops.constant_initializer(W2_init))        
	

        # Initialization of Spectral Normalized Recurrent Matrix (W2_bar)
        self._W2_bar = tf.get_variable("W2_bar", [self._hidden_size_ENRNN, self._hidden_size_ENRNN], \
                      initializer = init_ops.constant_initializer())


        # Initialization of Transition Recurrent Matrix (W3)
        if coupling == "coupled":
            self._W3 = tf.get_variable("W3", [self._hidden_size_ENRNN, self._hidden_size_scoRNN], \
                                       initializer = tf.glorot_uniform_initializer())
   
   
    @property
    def state_size(self):
        return self._hidden_size_scoRNN + self._hidden_size_ENRNN

    @property
    def output_size(self):
        return self._hidden_size_scoRNN + self._hidden_size_ENRNN

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ENRNNCell"):
            
            # Initialization of input matrix
            U_init = init_ops.random_uniform_initializer(-0.01, 0.01)
            U = vs.get_variable("U", [inputs.get_shape()[-1], \
                self._hidden_size_scoRNN + self._hidden_size_ENRNN], initializer= U_init)

	
	        # Initialization of bias
            bias = tf.get_variable("b", [self._hidden_size_scoRNN + self._hidden_size_ENRNN], \
	        initializer= init_ops.constant_initializer())

                       
            # Forward pass of graph
            recurrent1 = math_ops.matmul(state[:,:self._hidden_size_scoRNN], self._W1)
            recurrent2 = math_ops.matmul(state[:, self._hidden_size_scoRNN:], self._W2_bar)
            if self._coupling == "coupled":
                recurrent3 = math_ops.matmul(state[:, self._hidden_size_scoRNN:], self._W3)
                recurrent_combined = tf.concat([recurrent1 + recurrent3, recurrent2], axis=1)
            else:
                recurrent_combined = tf.concat([recurrent1, recurrent2], axis=1)
           
            res = math_ops.matmul(inputs, U) + recurrent_combined
            if self._activation_function == 'modrelu':
                output = tf.nn.relu(nn_ops.bias_add(tf.abs(res), bias))*(tf.sign(res))
            else:
                res_b = nn_ops.bias_add(res, bias)

                if self._activation_function != None :
                    # Activation function dictionary
                    activation_dict = {'relu'     : tf.nn.relu, 
                                       'tanh'     : tf.nn.tanh,
                                       'sigmoid' : tf.nn.sigmoid} 
                    output = activation_dict[self._activation_function](res_b)               
                else:
                    output = tf.nn.relu(res_b)

        return output, output

