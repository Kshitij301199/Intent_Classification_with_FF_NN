from typing import Tuple
import numpy as np
import numpy.typing as npt
from model.model_utils import softmax, relu, relu_prime

def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(pred, epsilon, 1 - epsilon)
    
    return -np.mean(np.sum(truth * np.log(y_pred),axis=0)) / len(truth)


class NeuralNetwork:
    """
    A simple neural network implementation with one hidden layer.
    
    This class represents a neural network with one hidden layer, ReLU activation, and softmax output.
    It uses cross-entropy loss for training. The network can be initialized with specific sizes for
    the input layer, hidden layer, and output layer.

    Parameters:
    - input_size (int): Number of input features.
    - hidden_size (int): Number of neurons in the hidden layer.
    - num_classes (int): Number of output classes.
    - random_seed (int, optional): Seed for random number generation. Default is 1.
    - learning_rate (float, optional): Learning rate for gradient descent. Default is 0.005.

    Attributes:
    - input_size (int): Number of input features.
    - hidden_size (int): Number of neurons in the hidden layer.
    - output_size (int): Number of output classes.
    - num_classes (int): Number of output classes.
    - learning_rate (float): Learning rate for gradient descent.
    - internal_weights (numpy.ndarray): Weights for the hidden layer.
    - output_weights (numpy.ndarray): Weights for the output layer.

    Methods:
    - forward(X: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        Forward pass with X as input matrix, returning the prediction Y_hat.

    - predict(X: npt.ArrayLike) -> npt.ArrayLike:
        Create a prediction matrix with `self.forward()`.

    - backward(X: npt.ArrayLike, Y: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        Backpropagation algorithm.

    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        random_seed: int = 1,
        learning_rate: float = 0.005,
    ):
        """
        Initialize neural network's weights and biases.
        
        Parameters:
        - input_size (int): Number of input features.
        - hidden_size (int): Number of neurons in the hidden layer.
        - num_classes (int): Number of output classes.
        - random_seed (int): Seed for random number generation.
        - learning_rate (float): Learning rate for gradient descent.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        np.random.seed(random_seed)
        self.internal_weights = np.random.uniform(low = -1.0,
                                         high = 1.0,
                                         size = (input_size,hidden_size),
                                         )
        self.output_weights = np.random.uniform(low = -1,
                                                high = 1,
                                                size = (hidden_size,num_classes)
                                                )
        print("------------------Model Initialised------------------")
        
    def forward(self,
                X: npt.ArrayLike
                )-> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Forward pass with X as input matrix, returning the self prediction Y_hat.
        
        Parameters:
        - X (npt.ArrayLike): Input matrix.

        Returns:
        Tuple containing:
        - X (npt.ArrayLike): Input matrix with bias term.
        - z_internal (npt.ArrayLike): Internal layer input.
        - a_internal (npt.ArrayLike): Internal layer output after ReLU activation.
        - z_output (npt.ArrayLike): Output layer input.
        - a_output (npt.ArrayLike): Output layer output after softmax activation.
        """
        try:
            bias = np.ones(shape=X.shape[1]).reshape(1,X.shape[1])
            X = np.append(bias,X,axis=0)
            # print(f"Shape of Input Matrix : {X.shape}")
            z_internal: npt.ArrayLike = np.matmul(self.internal_weights.T,X)
            # print(f"Shape of internal weights : {z_internal.shape}")
            a_internal: npt.ArrayLike = np.array([relu(node) for node in z_internal])
            # print(a_internal.shape)
            z_output: npt.ArrayLike = np.matmul(self.output_weights.T,a_internal)
            # print(f"Shape of output weights : {z_output.shape}")
            a_output: npt.ArrayLike = softmax(z_output)
            # print(a_output.shape)
            
        except IndexError:
            bias = [1]
            X = np.append(bias,X,axis=0)
            # print(f"Shape of Input Matrix : {X.shape}")
            z_internal: npt.ArrayLike = np.matmul(self.internal_weights.T,X)
            # print(f"Shape of internal weights : {z_internal.shape}")
            a_internal: npt.ArrayLike = relu(z_internal)
            # print(a_internal.shape)
            z_output: npt.ArrayLike = np.matmul(self.output_weights.T,a_internal)
            # print(f"Shape of output weights : {z_output.shape}")
            a_output: npt.ArrayLike = softmax(z_output)
            a_output = a_output.reshape(self.output_size,1)
            # print(a_output.shape)
        
        return X, z_internal, a_internal, z_output, a_output

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`.
        
        Parameters:
        - X (npt.ArrayLike): Input matrix.

        Returns:
        - npt.ArrayLike: Prediction matrix.
        """
        try:
            prediction = []
            for i in range(X.shape[1]):
                _, _, _, _, a_output = self.forward(X[:,i])
                prediction_int = np.zeros(shape=self.output_size)
                prediction_int[a_output.argmax()] = 1
                prediction.append(list(prediction_int))
            prediction = np.array(prediction).T
        except IndexError:
            _, _, _, _, a_output = self.forward(X)
            prediction = np.zeros(shape=(self.output_size,1))
            prediction[a_output.argmax()] = 1
        
        # print(prediction.shape)
        return prediction

    def backward(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike
    ) -> Tuple[float, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        
        Parameters:
        - X (npt.ArrayLike): Input matrix.
        - Y (npt.ArrayLike): Ground truth labels.

        Returns:
        Tuple containing:
        - loss (float): Loss value.
        - grad_hidden_layer (npt.ArrayLike): Gradient of weights for the hidden layer.
        - grad_output_layer (npt.ArrayLike): Gradient of weights for the output layer.
        """
        try:
            _ = X.shape[1]
            X_bias, z_internal, a_internal, _, a_output = self.forward(X)
            
            loss = compute_loss(a_output,Y)

            cross_entropy_delta = np.subtract(a_output, Y)  # Derivative of softmax with cross-entropy loss
            grad_output_layer = np.matmul(a_internal, cross_entropy_delta.T)
            grad_hidden_input = np.matmul(self.output_weights,cross_entropy_delta) * np.array([relu_prime(z) for z in z_internal.T]).T
            grad_hidden_layer = np.dot(X_bias,grad_hidden_input.T)
        
        except IndexError:
            X_bias, z_internal, a_internal, _, a_output = self.forward(X)
            Y = Y.reshape(self.output_size,1)
            z_internal = z_internal.reshape(self.hidden_size,1)
            a_internal = a_internal.reshape(self.hidden_size,1)
            a_output = a_output.reshape(Y.shape)

            loss = compute_loss(a_output,Y)

            cross_entropy_delta = np.subtract(a_output, Y)  # Derivative of softmax with cross-entropy loss
            grad_output_layer = np.matmul(a_internal, cross_entropy_delta.T)
            grad_hidden_input = np.matmul(self.output_weights,cross_entropy_delta) * relu_prime(z_internal).reshape(self.hidden_size,1)
            grad_hidden_layer = np.dot(X_bias.reshape(self.input_size,1),grad_hidden_input.reshape(self.hidden_size,1).T)
            
        return loss, grad_hidden_layer, grad_output_layer
    