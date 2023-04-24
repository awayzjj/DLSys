import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(label_filename, 'rb') as f:
        file_content = f.read()
        magic_number, items_num = struct.unpack('>ii', file_content[0:8])
        labels = struct.unpack(f'>{items_num}s', file_content[8:])[0]
    y = np.frombuffer(labels, dtype=np.uint8)

    with gzip.open(image_filename, 'rb') as f:
        file_content = f.read()
        magic_number, items_num, rows, columns = struct.unpack('>iiii', file_content[0:16])
        pixels = struct.unpack(f'>{items_num * rows * columns}s', file_content[16:])[0]
    X = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 28 * 28).copy().astype(np.float32)
    X /= 255
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # def softmax(x):
    #     return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    # return np.mean(-np.log(softmax(Z)[np.indices(y.shape)[0], y]))

    # Simple version!
    k = Z.shape[1]
    O = np.log(np.exp(Z).sum(axis=1))
    #y_onehot = np.eye(k, dtype=np.int8)[y]
    y_onehot = np.eye(k)[y]
    return (O - (Z * y_onehot).sum(axis=1)).mean()
    ### END YOUR CODE

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1)) # 防止溢出
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.
    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]
    k = theta.shape[1]
    for i in range(0, m, batch):
        X_batch = X[i:i + batch] # batch * n
        y_batch = y[i:i + batch] # batch
        y_pre = np.dot(X_batch, theta) # batch * n n * k -> batch * k
        Z = softmax(y_pre) # batch * k 
        gradient = np.dot(X_batch.T, Z - np.eye(k)[y_batch]) / batch # n * batch batch * k
        theta -= lr * gradient
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    for iter in range(iter_num):
        iter_x = X[iter * batch: (iter+1) * batch, :]
        iter_y = y[iter * batch: (iter+1) * batch]
        Z1 = np.matmul(iter_x, W1)
        relu_mask = Z1 > 0 
        relu_Z1 = Z1 * relu_mask
        Z2 = np.matmul(relu_Z1, W2)
        
        cross_entropy_grad = np.exp(Z2) / np.sum(np.exp(Z2), axis=1, keepdims=True)
        cross_entropy_grad[np.indices(iter_y.shape)[0], iter_y] -= 1
        # Assume we reduce mean
        cross_entropy_grad /= batch
        # Compute W1 W2 grad, use Matmul
        W2_grad = np.matmul(np.transpose(relu_Z1), cross_entropy_grad)
        relu_grad = np.matmul(cross_entropy_grad, np.transpose(W2)) * relu_mask 
        W1_grad = np.matmul(np.transpose(iter_x), relu_grad)
        # Update Parameter
        W1 -= lr * W1_grad
        W2 -= lr * W2_grad
        
   # m = X.shape[0]
   # k = W2.shape[1]
   # for i in range(0, m, batch):
   #   X_batch = X[i:i+batch]
   #   y_batch = y[i:i+batch]

   #   Z1 = np.maximum(0, np.dot(X_batch, W1)) # b x hd

   #   y_pre = np.dot(Z1, W2) # b x k
   #   # y_e = np.exp(y_pre) # b x k
   #   # Z = y_e / np.sum(y_e, axis=1).reshape(-1, 1)
   #   Z = softmax(y_pre)
   #   I = np.eye(k)[y_batch] # b x k
   #   G2 = Z - I # b x k

   #   G1 = (np.float32(Z1>0)) * (np.dot(G2, W2.T)) # b x hd

   #   W1[:,:] = W1[:,:] - lr * (np.dot(X_batch.T, G1)) / batch
   #   W2[:,:] = W2[:,:] - lr * (np.dot(Z1.T, G2)) / batch
      
    #m = X.shape[0]
    #for i in range(0, m, batch):
    #    X_batch = X[i:i + batch]
    #    y_batch = y[i:i + batch]
    #    Z1 = np.maximum(X_batch @ W1, 0) # m * n n * d -> m * d
    #    G2 = softmax(Z1 @ W2) - np.eye(k)[y_batch] # m * d d * k -> m * k
    #    #G1 = (np.maximum(Z1, 0) / Z1) * (G2 @ W2.T)
    #    G1 = (np.float32(Z1>0)) * (np.dot(G2, W2.T))
    #    gradient1 = X_batch.T @ G1 / m # n * m m *n -> n * n
    #    gradient2 = Z1.T @ G2 / m # d * m m * k -> d * k
    #    W1 -= lr * gradient1
    #    W2 -= lr * gradient2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")

    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")


    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100)
    
    #print("\nTraining two layer neural network w/ 400 hidden units")
    #train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr = 0.2)