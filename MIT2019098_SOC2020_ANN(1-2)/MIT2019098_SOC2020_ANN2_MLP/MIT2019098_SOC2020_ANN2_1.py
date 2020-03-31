# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:02:24 2020

@author: Ratnesh
"""
import numpy as np
import matplotlib.pyplot as plt

#initialising network with two hidden layers and 1 output layer
#n_inp = no. of nueorns in input layer
#n_hidden = no. of nuerons in each hidden layer
#n_outputs = no. of nuerons in output layer

np.random.seed(17)

def initialize_netw(n_inp, n_hidden, n_outputs):
    network = []
    hidden_layer1 = [{'weights':np.random.randn(n_inp + 1) -0.5} for i in range(n_hidden)]
    network.append(hidden_layer1)
    hidden_layer2 = [{'weights':np.random.randn(n_hidden + 1) -0.5} for i in range(n_hidden)]
    network.append(hidden_layer2)
    output_layer = [{'weights':np.random.randn(n_hidden + 1) - 0.5} for i in range(n_outputs)]
    network.append(output_layer)
    return network

#activate function
def activate(w,inp):
    activation = w[-1]
    for i in range(len(w) -1):
        activation += w[i] * inp[i]
    return activation

def transfer(activation):
    return 1.0/(1.0 + np.exp(-activation))

#for forward porpagation
def forw_prop(network,row):
    inp = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inp)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inp = new_inputs
    return inp

#derivative of sigmoid function
def derv(out):
    return out*(1.0 - out)

#back propagation of error
def back_prop_error(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if(i != len(network) -1):
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error +=(neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derv(neuron['output'])



#after back_propagation update weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    errors = []

    cons_lmt = 100
    cnt_adp = cons_lmt
    cnt_adp_sign = cons_lmt - 1
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forw_prop(network, row)
            expected = [row[-1]]
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            back_prop_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errors.append(sum_error)

        #checking for covergence point
        if(sum_error < 0.011):
            break
#here we are applying adaptive learning
        if(epoch != 0):
            if((len(errors)>1) and ((errors[-2] - errors[-1] > 0 and errors[-1] - sum_error < 0)
                or (errors[-2] - errors[-1] < 0 and errors[-1] - sum_error > 0))) :
                cnt_adp_sign -=1
            else:
                cnt_adp_sign = cons_lmt

            if(errors[-1] > 0 and sum_error > 0):
                cnt_adp -=1
            else:
                cnt_adp = cons_lmt

            if(cnt_adp == 0):
                cnt_adp = cons_lmt
                l_rate = l_rate + 0.005

            if(cnt_adp_sign == 0):
                cnt_adp_sign = cons_lmt
                l_rate = l_rate - 0.005


    plt.plot([i for i in range(len(errors))],errors)
    plt.xlabel("number of epochs")
    plt.ylabel("squared sum error")
    plt.ylim(0,max(errors))

#For creation of training Data
def create_train_data(gate):
    train_X1 = np.array([0,0,1,1])
    train_X2 = np.array([0,1,0,1])
#    temp = np.array([1,1,1,1])
    X = np.column_stack((train_X1,train_X2))
    if(gate=="AND"):
        return X,np.array([0,0,0,1])
    elif(gate=="OR"):
        return X,np.array([0,1,1,1])
    elif(gate=="NOR"):
        return X,np.array([1,0,0,0])
    elif(gate == "NAND"):
        return X,np.array([1,1,1,0])
    elif(gate == "EX-OR"):
        return X,np.array([0,1,1,0])
    else:
        return X,np.array([1,0,0,1])

def NOT(x):
    z = np.zeros(x.shape)
    for i,a in enumerate(x):
        if(a):
            z[i] = 0
        else:
            z[i] = 1
    return z

X,Y = create_train_data('EX-NOR')
network = initialize_netw(X.shape[1],3, 1)

train_network(network, np.column_stack((X,Y)), 0.3, 10000, 1)

#for layer in network:
#    print(layer)

def predict(network, row):
    outputs = forw_prop(network, row)
    return (outputs[0] > 0.5) * 1.0
    return outputs.index(max(outputs))

print("\nTESTING\n")
for row in np.column_stack((X,Y)):
    prediction = predict(network, row)
    print('Expected=%d, Predictedt=%d' % (row[-1], prediction))



