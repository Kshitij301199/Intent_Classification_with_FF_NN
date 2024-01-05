import argparse
import numpy as np

from utils import load_dataset
from model.model_utils import bag_of_words_matrix, labels_matrix
from model.ffnn import NeuralNetwork
from helper import batch_train, minibatch_train

DATA_PATH = './data/dataset.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Train feedforward neural network'
    )

    parser.add_argument(
        '--minibatch', dest='minibatch',
        help='Train feedforward neural network with mini-batch gradient descent/SGD',
        action='store_true'
    )

    parser.add_argument(
        '--train', dest='train',
        help='Turn on this flag when you are ready to train the model with backpropagation.',
        action='store_true'
    )

    args = parser.parse_args()

    sentences, intent, unique_intent = load_dataset(DATA_PATH)
    
    X = bag_of_words_matrix(sentences=sentences,COUNT_THRESHOLD=2)
    Y = labels_matrix(data=(intent,unique_intent))
    print(f"Input Matrix Shape: {X.shape}")
    print(f"Target Matrix Shape: {Y.shape}")
    
    model = NeuralNetwork(input_size = X.shape[0]+1,
                          hidden_size = 150,
                          num_classes = Y.shape[0],
                          learning_rate = 0.005)

    if not args.minibatch:
        print("Training FFNN using batch gradient descent...")
        model = batch_train(X, Y, model, train_flag = args.train)
    else:
        print("Training FFNN using mini-batch gradient descent...")
        model = minibatch_train(X, Y, model,BATCH_SIZE=64, train_flag=args.train)

if __name__ == "__main__":
    main()
