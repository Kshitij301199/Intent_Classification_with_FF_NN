import numpy as np
import matplotlib.pyplot as plt
from model.ffnn import compute_loss

def batch_train(X, Y, model, train_flag=False):
    """
    Perform batch training on the neural network model.

    Parameters:
    - X (npt.ArrayLike): Input matrix.
    - Y (npt.ArrayLike): Ground truth labels.
    - model (NeuralNetwork): Neural network model to be trained.
    - train_flag (bool, optional): If True, train the model; if False, only evaluate accuracy. Default is False.

    Returns:
    - NeuralNetwork: Updated neural network model.

    Note:
    This function prints the accuracy, loss, and creates a plot of the cost function and accuracy over iterations.

    """
    NUM_EXAMPLES = X.shape[1]
    prediction = model.predict(X)
    hits = np.sum([np.array_equal(prediction[:,i],Y[:,i]) for i in range(X.shape[1])])
    print(f"Accuracy of model : {hits*100/X.shape[1]} %")
    
    if train_flag:
        loss_list = []
        batch_loss_list = []
        accuracy_list = []
        for i in range(NUM_EXAMPLES):
            loss, grad_hidden_layer, grad_output_layer = model.backward(X[:,i],Y[:,i])
            model.internal_weights -= model.learning_rate * grad_hidden_layer
            model.output_weights -= model.learning_rate * grad_output_layer
            batch_loss_list.append(loss)
            # loss_list.append(loss)
            if i%25 == 0:
                print(f"Inputs processed : {i}")
                prediction = model.predict(X)
                hits = np.sum([np.array_equal(prediction[:,i],Y[:,i]) for i in range(NUM_EXAMPLES)])
                loss_list.append(compute_loss(prediction,Y))
                accuracy_list.append(hits*100/1000)
                print(f"Current loss : {compute_loss(prediction,Y)} \tCurrent Accuracy : {hits*100/NUM_EXAMPLES} %")
                print(f"Batch loss : {loss}")
                    
        prediction = model.predict(X)
        hits = np.sum([np.array_equal(prediction[:,i],Y[:,i]) for i in range(X.shape[1])])
        print(f"Accuracy of model : {hits*100/X.shape[1]} %")
        
        plt.style.use("seaborn")
        fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(15,10),sharex=True)
        ax[0].plot(range(1,X.shape[1]+1,25),loss_list,"k-",linewidth=2,label = "Data Loss")
        ax[0].plot(range(1,X.shape[1]+1),batch_loss_list,'k--',linewidth=1,label = "Batch Loss")
        ax[1].plot(range(1,X.shape[1]+1,25),accuracy_list,"r-",linewidth=2)
        plt.xlabel("Iteration",fontdict={"size":14,"weight":"bold"})
        ax[0].set_ylabel("Cost Function",fontdict={"size":14,"weight":"bold"})
        ax[1].set_ylabel("Accuracy",fontdict={"size":14,"weight":"bold"})
        ax[1].set_ylim(0,100)
        ax[0].legend(loc="best")
        plt.xticks(range(X.shape[1]+1)[::100])
        fig.tight_layout()
        fig.savefig("./images/Cost_Function_Training.jpeg",dpi=300)
        print("Image saved at : ./images/Cost_Function_Training.jpeg")
        
        return model


def minibatch_train(X, Y, model, BATCH_SIZE:int = 64, train_flag = False):
    """
    Perform minibatch training on the neural network model.

    Parameters:
    - X (npt.ArrayLike): Input matrix.
    - Y (npt.ArrayLike): Ground truth labels.
    - model (NeuralNetwork): Neural network model to be trained.
    - BATCH_SIZE (int, optional): Size of minibatches. Default is 64.
    - train_flag (bool, optional): If True, train the model; if False, only evaluate accuracy. Default is False.

    Returns:
    - NeuralNetwork: Updated neural network model.

    Note:
    This function prints the accuracy, loss, and creates a plot of the cost function and accuracy over iterations.

    """
    NUM_EXAMPLES = X.shape[1]
    prediction = model.predict(X)
    hits = np.sum([np.array_equal(prediction[:,i],Y[:,i]) for i in range(NUM_EXAMPLES)])
    print(f"Accuracy of model : {hits*100/NUM_EXAMPLES} %")
    
    if train_flag:
        loss_list = []
        batch_loss_list = []
        accuracy_list = []
        i = 0
        while i < NUM_EXAMPLES:
            loss, grad_hidden_layer, grad_output_layer = model.backward(X[:,i:i+BATCH_SIZE],
                                                                        Y[:,i:i+BATCH_SIZE])
            model.internal_weights -= model.learning_rate * grad_hidden_layer
            model.output_weights -= model.learning_rate * grad_output_layer
            batch_loss_list.append(loss)
            print(f"Inputs processed : {i}")
            print(f"Batch loss : {loss}")
            prediction = model.predict(X)
            hits = np.sum([np.array_equal(prediction[:,i],Y[:,i]) for i in range(NUM_EXAMPLES)])
            loss_list.append(compute_loss(prediction,Y))
            accuracy_list.append(hits*100/1000)
            print(f"Current loss : {compute_loss(prediction,Y)} \tCurrent Accuracy : {hits*100/NUM_EXAMPLES} %")
            i += BATCH_SIZE
            
    plt.style.use("seaborn")
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(15,10),sharex=True)
    ax[0].plot(range(1,X.shape[1]+1,BATCH_SIZE),loss_list,"k-",linewidth=2,label = "Data Loss")
    ax[0].plot(range(1,X.shape[1]+1,BATCH_SIZE),batch_loss_list,'k--',
               linewidth=1,label = "Batch Loss")
    ax[1].plot(range(1,X.shape[1]+1,BATCH_SIZE),accuracy_list,"r-",linewidth=2)
    plt.xlabel("Iteration",fontdict={"size":14,"weight":"bold"})
    ax[0].set_ylabel("Cost Function",fontdict={"size":14,"weight":"bold"})
    ax[1].set_ylabel("Accuracy",fontdict={"size":14,"weight":"bold"})
    ax[1].set_ylim(0,100)
    ax[0].legend(loc="best")
    plt.xticks(range(X.shape[1]+1)[::100])
    fig.tight_layout()
    fig.savefig("./images/MiniBatch_Cost_Function_Training.jpeg",dpi=300)
    print("Image saved at : ./images/MiniBatch_Cost_Function_Training.jpeg")
    
    return model
