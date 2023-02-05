# models.py

from turtle import hideturtle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.utils.data import Dataset, DataLoader


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, word_embeddings):

        # TODO: Your code here!
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)
        self.word_embeddings = word_embeddings
        # raise NotImplementedError
    
    def forward(self, x) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.

        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """
        # TODO: Your code here!
        out = self.linear2(torch.relu(self.linear1(x)))
        return F.log_softmax(out, dim=1)
        raise NotImplementedError


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model: int):
        # TODO: Your code here!
        self.model = model
        # raise NotImplementedError

    def predict(self, ex_words: List[str]) -> int:
        # TODO: Your code here!
        self.model.eval()
        word_embeddings = self.model.word_embeddings 
        embedded_ex = np.array([word_embeddings.get_embedding(word) for word in ex_words])
        average_embedding = np.array([np.mean(embedded_ex, axis=0)])
        average_embedding = torch.from_numpy(average_embedding).float()
        probabilities = self.model.forward(average_embedding)
        return torch.argmax(probabilities).item()
        return NotImplementedError


def get_embedded_data(word_embeddings: WordEmbeddings, exs: List[SentimentExample]):
    average_vectors = []
    labels = []
    for ex in exs:
        embedded_ex = np.array([word_embeddings.get_embedding(word) for word in ex.words])
        average_embedding = np.mean(embedded_ex, axis=0)
        average_vectors.append(average_embedding)
        labels.append(ex.label)
    average_vectors = np.array(average_vectors)
    labels = np.array(labels)
    # print(average_vectors.shape, labels.shape)
    return average_vectors, labels

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # TODO: Your code here!
    X_train, y_train = get_embedded_data(word_embeddings, train_exs)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    X_dev, y_dev = get_embedded_data(word_embeddings, dev_exs)
    X_dev = torch.from_numpy(X_dev).float()
    y_dev = torch.from_numpy(y_dev).long()

    input_dim = int(word_embeddings.get_embedding_length())

    model = NeuralNet(input_dim, args.hidden_size, word_embeddings)

    criterion = nn.NLLLoss()

    milestones = [int(0.25*args.num_epochs),int(0.5*args.num_epochs)]
    gamma = 0.1
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    loss_val_ls = []

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model(X_train)
        loss = criterion(y_hat,y_train)
        loss.backward()
        optimizer.step()
        prediction_matches = y_hat.max(dim=1).indices.eq(y_train)
        accuracy = prediction_matches.float().mean()
        print("train accuracy = ", np.round(accuracy.item(), 3))

        model.eval()
        with torch.no_grad():
            y_hat = model(X_dev)
            eval_loss = criterion(y_hat, y_dev) 
            # print("Evaluation Loss:", eval_loss.item())
            prediction_matches = y_hat.max(dim=1).indices.eq(y_dev)
            accuracy = prediction_matches.float().mean()
            print("valid accuracy = ", np.round(accuracy.item(), 3))
        
        # scheduler.step()

    return NeuralSentimentClassifier(model)

    raise NotImplementedError

