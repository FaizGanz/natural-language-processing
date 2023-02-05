# Homework 1 Instructions

NLP with Representation Learning, Fall 2022

Name: Faiz Andrea Ganz

Net ID: fag277

## Part 1: N-Gram Model

### Part 1a

```
Vocab size: 33175
Train Perplexity: 8.107
Valid Perplexity: 10121.316
```

## Part 2: N-Gram Model with Additive Smoothing

### Part 2a

Sparsity in our training set data matrix causes the probability of unseen n-grams in our test dataset to drop to zero.

### Part 2b

```
Vocab size: 33175
Train Perplexity: 116.398
Valid Perplexity: 2844.227
```

### PART 2c

--n=2 --delta=0.00005
Vocab size: 33175
Train Perplexity: 79.204
Valid Perplexity: 508.939

--n=2 --delta=0.0005
Vocab size: 33175
Train Perplexity: 90.228
Valid Perplexity: 440.781

--n=2 --delta=0.005
Vocab size: 33175
Train Perplexity: 138.787
Valid Perplexity: 447.895

--n=2 --delta=0.05
Vocab size: 33175
Train Perplexity: 331.400
Valid Perplexity: 663.045

--n=2 --delta=0.5
Vocab size: 33175
Train Perplexity: 1133.768
Valid Perplexity: 1411.721

--n=3 --delta=0.00005
Vocab size: 33175
Train Perplexity: 11.219
Valid Perplexity: 2981.892

--n=3 --delta=0.0005
Vocab size: 33175
Train Perplexity: 26.768
Valid Perplexity: 2421.402

--n=3 --delta=0.005
Vocab size: 33175
Train Perplexity: 116.398
Valid Perplexity: 2844.227

--n=3 --delta=0.05
Vocab size: 33175
Train Perplexity: 733.721
Valid Perplexity: 4648.274

--n=3 --delta=0.5
Vocab size: 33175
Train Perplexity: 4701.752
Valid Perplexity: 9198.110

The validation perplexity of a bigram model is substantially better than the one of a trigram model across all values of delta. The trigram model seems to significantly overfit the data. The best value for the validation perplexity is given by parameters (n=2, delta=0.0005)

## Part 3: N-Gram Model with Interpolation Smoothing

### Part 3a

```
Vocab size: 33175
Train Perplexity: 17.596
Valid Perplexity: 293.566
```

### Part 3b

**TODO**: Report validation perplexity for different lambda values and select best lambdas.

--lambda1=0.7 --lambda2=0.2 --lambda3=0.1
Train Perplexity: 42.858
Valid Perplexity: 344.115

--lambda1=0.5 --lambda2=0.3 --lambda3=0.2
Train Perplexity: 25.513
Valid Perplexity: 298.598

--lambda1=0.25 --lambda2=0.5 --lambda3=0.25
Train Perplexity: 19.739
Valid Perplexity: 282.367

--lambda1=0.1 --lambda2=0.8 --lambda3=0.1
Train Perplexity: 28.011
Valid Perplexity: 304.102

--lambda1=0.2 --lambda2=0.3 --lambda3=0.5
Train Perplexity: 13.169
Valid Perplexity: 319.886

--lambda1=0.1 --lambda2=0.2 --lambda3=0.7
Train Perplexity: 10.400
Valid Perplexity: 401.368

Best Lambdas: --lambda1=0.25 --lambda2=0.5 --lambda3=0.25

## Part 4: Backoff

### Part 4a

```
Vocab size: 33175
Train Perplexity: 8.107
Valid Perplexity: 142.995
```

## Part 5: Test Set Evaluation

### Part 5a

**TODO**: Test set perplexity for each model type. Indicate best model.

vanilla --n=2 --test
Test Perplexity: 513.943

additive --n=2 --delta=0.0005 --test
Test Perplexity: 412.149

interpolation --lambda1=0.25 --lambda2=0.5 --lambda3=0.25 --test
Test Perplexity: 267.175

backoff --test
Test Perplexity: 136.922

### Part 6: A Taste of Neural Networks

**TODO**: Report train and development set accuracy

--num_epochs=1000 --lr=0.01

=====Train Accuracy=====
Accuracy: 5061 / 6920 = 0.731358;
Precision (fraction of predicted positives that are correct): 2679 / 3607 = 0.742722;
Recall (fraction of true positives predicted correctly): 2679 / 3610 = 0.742105;
F1 (harmonic mean of precision and recall): 0.742414;

=====Dev Accuracy=====
Accuracy: 628 / 872 = 0.720183;
Precision (fraction of predicted positives that are correct): 345 / 490 = 0.704082;
Recall (fraction of true positives predicted correctly): 345 / 444 = 0.777027;
F1 (harmonic mean of precision and recall): 0.738758;

Time for training and evaluation: 4.30 seconds