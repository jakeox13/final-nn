# TODO: import dependencies and write unit tests below
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nn import *

def test_single_forward():
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}],
                        lr=0.01, seed=42, batch_size=10, epochs=100, loss_function="binary_cross_entropy")
    W_curr = np.array([[1, 2, 3], [4, 5, 6]])
    b_curr = np.array([[0.1], [0.2]])
    A_prev = np.array([[0.5], [0.6], [0.7]]).T
    activation = 'relu'

    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    #Expected values
    a_curr_expt=np.array([3.9, 9.4])
    z_curr_expt=np.array([3.9, 9.4])

    # Check correct implimentation
    assert np.allclose(A_curr,a_curr_expt,rtol=1e-03)
    assert np.allclose(Z_curr,z_curr_expt,rtol=1e-03)
    pass

def test_forward():

    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    W1 = np.array([[1, 2, 3], [4, 5, 6]])
    b1 = np.array([[0.1], [0.2]])
    W2 = np.array([[0.7, 0.8], [0.9, 1.0]])
    b2 = np.array([[0.3], [0.4]])
    arch = [{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'}, {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}]

    # Initialize NeuralNetwork instance
    nn = NeuralNetwork(nn_arch=arch, lr=0.01, seed=42, batch_size=32, epochs=10, loss_function='binary_cross_entropy')

    # Set parameters manually for testing
    nn._param_dict['W1'] = W1
    nn._param_dict['b1'] = b1
    nn._param_dict['W2'] = W2
    nn._param_dict['b2'] = b2

    # Call forward method
    output, cache = nn.forward(X)

    # Check out bout shape
    assert output.shape==(2,2)

    # Check that cache exists

    assert cache != None
    pass

def test_single_backprop():
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}],
                        lr=0.01, seed=42, batch_size=10, epochs=100, loss_function="binary_cross_entropy")
    W_curr = np.array([[1, 2, 3],])
    b_curr = np.array([[0.1]])
    Z_curr = np.array([[3.9]])
    A_prev = np.array([[0.5], [0.6], [0.7]]).T
    dA_curr = np.array([[0.1]])
    activation_curr = 'relu'

    

    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
    assert dA_prev.shape== (3,1)
    assert dW_curr.shape== (1,3)
    assert db_curr.shape == (1,1)
    pass

def test_predict():
    # Read in and process sample data
    pos=read_text_file("data/rap1-lieb-positives.txt")
    neg=read_fasta_file("data/yeast-upstream-1k-negative.fa")
    sample_data,sample_labels=sample_seqs(pos+neg, labels=([1]*len(pos))+([0]*len(neg)), total_samples=5000)
    sample_encoded=one_hot_encode_seqs(sample_data)
    # Split off Final test set
    X_train, X_test, y_train, y_test = train_test_split(np.array(sample_encoded), np.array(sample_labels), test_size=0.2, random_state=123)

    # Split intial train into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)  
    test=NeuralNetwork([{'input_dim': 17*4, 'output_dim': 17*4, 'activation': 'relu'}, {'input_dim': 17*4, 'output_dim': 34, 'activation': 'relu'}, {'input_dim': 34, 'output_dim': 17, 'activation': 'relu'},{'input_dim': 17, 'output_dim': 1, 'activation': 'sigmoid'}],lr=.01,seed=13,batch_size=10,epochs=700,loss_function="binary_cross_entropy")
    per_epoch_loss_train,per_epoch_loss_val=test.fit(X_train,y_train,X_val,y_val)
    y_pred=test.predict(X_val)
	
    
    # Check that my predictions are the same length as the number of inputs and width one since it is binary classifaction
    assert y_pred.shape == (len(X_val),1)
    pass

def test_binary_cross_entropy():
    # Read in and process sample data
    pos=read_text_file("data/rap1-lieb-positives.txt")
    neg=read_fasta_file("data/yeast-upstream-1k-negative.fa")
    sample_data,sample_labels=sample_seqs(pos+neg, labels=([1]*len(pos))+([0]*len(neg)), total_samples=5000)
    sample_encoded=one_hot_encode_seqs(sample_data)
    # Split off Final test set
    X_train, X_test, y_train, y_test = train_test_split(np.array(sample_encoded), np.array(sample_labels), test_size=0.2, random_state=123)

    # Split intial train into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)  
    test=NeuralNetwork([{'input_dim': 17*4, 'output_dim': 17*4, 'activation': 'relu'}, {'input_dim': 17*4, 'output_dim': 34, 'activation': 'relu'}, {'input_dim': 34, 'output_dim': 17, 'activation': 'relu'},{'input_dim': 17, 'output_dim': 1, 'activation': 'sigmoid'}],lr=.01,seed=13,batch_size=10,epochs=700,loss_function="binary_cross_entropy")
    per_epoch_loss_train,per_epoch_loss_val=test.fit(X_train,y_train,X_val,y_val)
    y_pred=test.predict(X_val)
	
    
    # Check that my predition matchs Sklearn
    assert sklearn.metrics.log_loss(y_val,y_pred) == test._binary_cross_entropy(y_val,y_pred)

    pass

def test_binary_cross_entropy_backprop():
    # Set up the NeuralNetwork with dummy parameters
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}],
                        lr=0.01, seed=42, batch_size=10, epochs=100, loss_function="binary_cross_entropy")

    # Set up test data
    y = np.array([[1], [0], [1]]) 
    y_hat = np.array([[0.9], [0.3], [0.7]]) 

    dA_binary= nn._binary_cross_entropy_backprop(y, y_hat)
    # chcek for right shape
    assert dA_binary.shape ==(3,1)
    manual_calc=np.array([-1/0.9,1/0.7,-1/0.7]).reshape(-1, 1)
    assert np.allclose(dA_binary,manual_calc,rtol=1e-03)
   
    pass

def test_mean_squared_error():
    digits = load_digits()
    # Split off Final test set
    X_train, X_test, y_train, y_test = train_test_split(digits["data"], digits["data"], test_size=0.2, random_state=13)

    # Split intial train into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, X_train, test_size=0.2, random_state=13)
    auto_encoder=NeuralNetwork([{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],lr=.001,seed=13,batch_size=10,epochs=100,loss_function="mean_squared_error")
    per_epoch_loss_train,per_epoch_loss_val=auto_encoder.fit(X_train,y_train,X_val,y_val)
    y_pred=auto_encoder.predict(X_val)
	
    # Check that my predition matchs Sklearn
    assert sklearn.metrics.mean_squared_error(y_val,y_pred) == auto_encoder._mean_squared_error(y_val,y_pred)
    pass

def test_mean_squared_error_backprop():
    # Set up the NeuralNetwork with dummy parameters
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 1, 'activation': 'relu'}],
                        lr=0.01, seed=42, batch_size=10, epochs=100, loss_function="binary_cross_entropy")

    # Set up test data
    y = np.array([[1], [0], [1]]) 
    y_hat = np.array([[0.9], [0.3], [0.7]]) 

    dA_mean = nn._mean_squared_error_backprop(y, y_hat)

     # chcek for right shape
    assert dA_mean.shape ==(3,1)
    #Close enough values
    manual_calc=np.array([-1/15,0.2,-0.2]).reshape(-1, 1)
    assert np.allclose(dA_mean,manual_calc,rtol=1e-03)
    
    pass

def test_sample_seqs():
    
    seqs = ["AGA", "GGG", "ATG", "GGC", "ATA"]
    labels = [True, False, True, False, True]  # Example labels (True for positive, False for negative)
    total_samples = 10
    
    # Sample seqs
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels, total_samples)
    # Check that output is the right length
    assert len(sampled_seqs)==10 and len(sampled_labels)==10
    # Check for balanced classes
    assert sum(sampled_labels*1)==5

    pass

def test_one_hot_encode_seqs():
    # Check Correct implmentation
    assert one_hot_encode_seqs(["AGA"]) == [[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
    
    # Check error raised
    with pytest.raises(ValueError, match= r"Invaild Base X"):
        one_hot_encode_seqs("XGA")
    pass