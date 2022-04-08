import numpy as np


def sigmoid(z):
    A = 1/(1+np.exp(-z))
    cache = z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    z = cache
    s = 1/(1+np.exp(-z))

    dZ = dA * s *( 1- s)
    assert(dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    z = cache

    dZ = np.array(dA, copy = True)
    #when z <= 0, so dz = 0
    #z > 0 dz = 1
    dZ[Z<=0] = 0
    assert(dZ.shape ==  z.shape)
    return dZ

def tanh(dA, cache):
    z = cache
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    dZ = dA*(1-A*A)
    assert(dZ.shape == z.shape)
    return dZ

def initialize_parameters_deepnn(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])* np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache =(A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b) #linear cache = (A_prev, W, b)
        A, activation_cache = sigmoid(Z) #activation_cache = Z
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    #X.shape =(input_size, number of examples)
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in NN
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
    AL, cache =  linear_activation_forward(A, parameters['W' + str(L)], parameters['W' + str(L)], activation = 'relu')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    #len(caches) = L (1--> L layer)
    return AL, caches

def compute_cost(AL, Y):

    #Y.shape = (1, number_of_examples)
    m = Y.shape[1]
    cost = (1./m)*(-np.dot(Y, np.log(AL).T)-np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost) # make sure that cost'shape is that we expect (([[17]] into 17))
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    #dZ = gradient of the cost with respect to the linear output
    #cache = (A_prev, W, b) coming from the forward propagation in the current layer

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    #caches = list of cache
    #cache = (linear cache, activation cache) for current layer
    #linear cache = (A_prev, W, b)
    #activation_cache = Z
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    #L = -(y*log(y-) + (1-y)*log(1-y-))
    # y- = AL
    #dL/dAL = dL/dy- = -(y/AL - (1-y)/(1-AL))
    dAL = -(np.divide(Y,AL) - np.divide(1-Y, 1- AL))
    current_cache = caches[L-1]
    grads['dA' + str(L)], grads['dW'+ str(L), grads['db' + str(L)]] = linear_activation_backward(dAL, current_cache, activation = 'sigmoid')

    for l in reversed(L-1):
        current_cache = caches[l]
        dA_prev_temp, dW_prev_temp, db_prev_temp = linear_activation_backward(grad['dA' + str(l+2)], current_cache, activation = 'relu')
        grads['dA' + str(l+1)] = dA_prev_temp
        grads['dA' + str(l+1)] = dW_prev_temp
        grads['dA' + str(l+1)] = db_prev_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]
    return parameters

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) //2
    p = np.zeros((1,m))

    AL, caches = L_model_forward(X, parameters)
    for i in range(0, AL.shape[1]):
        if AL[0, i] > 0.5:
            p[0,i] = 1
        else:
            p[0, i] = 0
    print('Accuracy: ' + srt(np.sum((p == y) / m)))
    return p
