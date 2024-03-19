# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()
        self._forward_activation_dict= {"relu":self._relu , "sigmoid":self._sigmoid}
        self._backprop_activation_dict= {"relu":self._relu_backprop , "sigmoid":self._sigmoid_backprop}
        self._loss_func_dict={"binary_cross_entropy": self._binary_cross_entropy, "mean_squared_error":self._mean_squared_error}
        

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        
        # Calculate Z
        #Z_curr=W_curr*A_prev+b_curr
        Z_curr = np.dot(W_curr, A_prev.T).T + b_curr.T
        # Pull the activation function
        active=self._forward_activation_dict[activation]
        
        # Calculate Current activation
        A_curr=active(Z_curr)
        return (A_curr,Z_curr)
        pass

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Check that X matches the input dimensions of first layer
        if X.shape[1] != self.arch[0]['input_dim']:
            raise ValueError("Input must match input dimensions for first layer")
        act_prev=X
        #Save inputs as intial activations
        cache={"A0":act_prev}
        # Pass through each layer
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            #print("Forward layer {}".format(layer_idx))
            # Calculate activation for single layer
            a,z=self._single_forward(self._param_dict['W' + str(layer_idx)],self._param_dict['b' + str(layer_idx)],A_prev=act_prev,activation=layer['activation'])
            
            # Cache values for backpropigation
            cache['A' + str(layer_idx)]= a
            cache['Z' + str(layer_idx)]= z
            
            #Update previous activation for next layer
            act_prev=a

        # Outout will be final activation
        output=a

        return output,cache
        pass

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Pull the activation function
        back_prop=self._backprop_activation_dict[activation_curr]
        
        # Calculate dZ
        dZ=back_prop( dA= dA_curr, Z= Z_curr)

        # Adjust shape
        dZ=dZ.T

        m = A_prev.shape[1]  # Number of examples

        dA_prev = np.dot(W_curr.T, dZ)
        dW_curr = np.dot(dZ, A_prev) / m
        db_curr = np.sum(dZ, axis=1, keepdims=True) / m

        return (dA_prev,dW_curr,db_curr)

        pass

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Create grad_dict
        grad_dict={}
        self.gradient_logs = {}
        # Calculate intial error derivatives
        if self._loss_func == "binary_cross_entropy":
            dA=self._binary_cross_entropy_backprop(y,y_hat)
        else:
            dA=self._mean_squared_error_backprop(y,y_hat)
        # Loop Through backprop layers
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1
            #print("Back prop layer {}".format(layer_idx))
            # Calculate backprop 
            da,dw,db =self._single_backprop(self._param_dict['W' + str(layer_idx)],self._param_dict['b' + str(layer_idx)],A_prev=cache['A' + str(layer_idx-1)],Z_curr=cache['Z' + str(layer_idx)],dA_curr=dA,activation_curr=layer['activation'])
            
            # Strore
            grad_dict['dA' + str(layer_idx)]= da
            grad_dict['dW' + str(layer_idx)]= dw
            grad_dict['dB' + str(layer_idx)]= db
            self.gradient_logs['dW' + str(layer_idx)] = grad_dict['dW' + str(layer_idx)]  # Store gradients
            
            #Update previous activation for next layer
            dA=da.T
        return grad_dict
        pass

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            

            # Update weights
            self._param_dict['W' + str(layer_idx)]=self._param_dict['W' + str(layer_idx)]- self._lr*(1/(self._batch_size)*(grad_dict['dW' + str(layer_idx)]))
            self._param_dict['b' + str(layer_idx)]=self._param_dict['b' + str(layer_idx)]-self._lr*(1/(self._batch_size)*(grad_dict['dB' + str(layer_idx)]))
    
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        #Set up results
        per_epoch_loss_train=[]
        per_epoch_loss_val=[]

        #Set up loss function
        lf=self._loss_func_dict[self._loss_func]
        # Repeat until convergence or maximum iterations reached
        epoch=0
        while epoch < self._epochs:


            # Create batches
            num_batches = int(X_train.shape[0] / self._batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            

            # Iterate through batches
            for X_train, y_train in zip(X_batch, y_batch):
                # Forward and back propgation
                y_pred,chace = self.forward(X_train)
                grad_dict=self.backprop(y_train,y_pred,chace)

                # Update weights
                self._update_params(grad_dict)

            # Calculate loss
            per_epoch_loss_train.append(lf(y_train, self.predict(X_train)))
            per_epoch_loss_val.append(lf(y_val, self.predict(X_val)))
            # Update iteration
            epoch += 1

        return per_epoch_loss_train,per_epoch_loss_val
        pass

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        output,chace=self.forward(X)
        y_hat = output
        return y_hat
        pass

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))
        pass

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        # Impliment the partial derivative of a sigmoid
        dZ = dA * Z * (1 - Z)
        return dZ

        pass

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)
        pass

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #print(dA.shape, Z.shape)
        # Relu derivative
        dZ = np.where(Z > 0, dA, 0)
        
        return dZ
        pass

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
         # Calculate binary cross-entropy loss for each sample
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) 
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # Take the average over the mini-batch
        loss = np.mean(losses)
        return loss
        pass

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Prevent divide by zero
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) 
        y = y.reshape(-1, 1)  
        # Caclulate partial derivatives
        dA = -(y / y_hat) + ((1 - y) / (1 - y_hat))

        return(dA)
        pass

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # set up empty array
        #loss=[]

        # Loop through all values
        #for i in range(len(y)):
            #Calculate biniary corss entorpy loss using formula
            #loss.append((y-y_hat)**2)
        
        # return the averge of all loses
        # return np.mean(loss)
    
         # Calculate squared error for each sample
        squared_errors = np.square(y - y_hat)
        
        # Calculate the mean squared error for the mini-batch
        loss = np.mean(squared_errors)
         
        return loss
        pass

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        n = len(y)
        dA = (2/n) * (y_hat - y)
        return dA
        pass

    def plot_gradients(self):
        # Plot gradients over time
        # Example:
        import matplotlib.pyplot as plt

        for key, value in self.gradient_logs.items():
            plt.plot(value, label=key)
        
        plt.xlabel('Epochs')
        plt.ylabel('Gradient Magnitude')
        plt.legend()
        plt.show()
        pass