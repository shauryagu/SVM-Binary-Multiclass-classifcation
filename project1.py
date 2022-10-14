"""EECS 445 - Fall 2022.

Project 1
"""
import numpy
import pandas as pd
import numpy as np
import itertools
import string

from scipy.stats import loguniform
import sklearn.svm
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import copy


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    input_string = input_string.lower()
    for c in range(len(input_string)):
        if input_string[c] in string.punctuation:
            input_string = input_string.replace(input_string[c], " ")
    return input_string.split()



def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    word = 0
    for x in df["text"]:
        comment = extract_word(x)
        for y in comment:
            if y not in word_dict:
                word_dict[y] = word
                word = word+1
    print("num words: ", len(word_dict))
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    # TODO: Implement this function
    #  CHANGED FOR PART 6
    data = [x for x in df['text']]
    # for x in range(number_of_reviews):
    #     words = extract_word(df["text"][x])
    #     string = ""
    #     for y in words:
    #         if y in word_dict:
    #             string += (y+' ')
    #     data[x] = string
    vect = TfidfVectorizer(stop_words=stopwords.words('english'))  # from here down, part 6
    feature_matrix = vect.fit_transform(data).toarray(order='C')
    print(feature_matrix.shape)
    return feature_matrix, vect




def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[1,-1]).ravel()
    if metric == "specificity" or metric == "sensitivity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred, )
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        return tp/(tp+fn)
    elif metric == "specificity":
        return tn/(tn+fp)




def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    # Put the performance of the model on each fold in the scores array
    scores = []
    print(X[0])
    skf = StratifiedKFold(n_splits=k)
    for train_index, test_index in skf.split(X, y):
        x_split = np.empty(shape=(len(train_index), X.shape[1]))
        y_split = np.empty(shape=(len(train_index), 1))
        x_val = np.empty(shape=(len(test_index), X.shape[1]))
        y_val = np.empty(shape=(len(test_index), 1))
        for ind in range(len(train_index)):
            x_split[ind] = X[train_index[ind]]
            y_split[ind] = y[train_index[ind]]
        for ind in range(len(test_index)):
            x_val[ind] = X[test_index[ind]]
            y_val[ind] = y[test_index[ind]]
        clf.fit(x_split, y_split)
        if metric != "auroc":
            scores.append(performance(y_val, clf.predict(x_val), metric=metric))
        else:
            scores.append(performance(y_val, clf.decision_function(x_val), metric="auroc"))
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with 
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    best_acc = [-1, -1]
    for c in C_range:
        clf = OneVsRestClassifier(LinearSVC(penalty=penalty, loss=loss, C=c,
                                            dual=dual, random_state=445, class_weight='balanced'))
        # CHANGED FOR PART 6
        acc = cv_performance(clf, X, y, k, metric)
        print(acc, c)
        if acc > best_acc[0]:
            best_acc[0] = acc
            best_acc[1] = c
    print(best_acc)
    return best_acc[1]



def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = sklearn.svm.LinearSVC(penalty=penalty, loss=loss, C=c, dual=dual, random_state=445)
        clf.fit(X, y)
        norm0.append(np.linalg.norm(clf.coef_[0], ord=0))
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM 
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    best_acc = 0
    # grid search
    print(param_range.shape)
    for p in param_range:
        print(p)
        clf = sklearn.svm.SVC(kernel="poly", degree=2, C=p[0], coef0=p[1], gamma='auto', class_weight='balanced')
        acc = cv_performance(clf, X, y, k, metric)
        if acc > best_acc:
            best_C_val = p[0]
            best_r_val = p[1]
            best_acc = acc
    print(best_acc)
    return best_C_val, best_r_val




def main():
    Read binary data
    #NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #     IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )


    # TODO: Questions 3, 4, 5
    q3
    words = extract_word("It's a test sentence! Does it look CORRECT?")
    print(words)

    metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
    clf = sklearn.svm.LinearSVC(penalty="l2", loss="hinge", C=.1, dual=True, random_state=445)
    clf.fit(X_train, Y_train)
    # p = performance(Y_test, clf.predict(X_test), metric="specificity")
    # print(p)

    # part 4d
    print(dictionary_binary)
    coef_dict = {x:clf.coef_[0, x] for x in range(len(dictionary_binary))}
    sorted_coef = sorted(coef_dict.items(), key=lambda kv:(kv[1], kv[0]))
    plot_vals = sorted_coef[0:5] + sorted_coef[-5:]
    npcoef = []
    words = []
    for x in range(10):
        npcoef.append(plot_vals[x][1])
        words.append(list(dictionary_binary.keys())[plot_vals[x][0]])
    print(npcoef, words)
    fig = plt.figure(figsize=(10,5))
    plt.bar(words, npcoef)
    plt.savefig('4dbar.png')
    plt.close()

    select_param_linear(X_train, Y_train, 5, "auroc", [.001,.01,.1,1], 'squared_hinge', 'l1', False)
    clf = sklearn.svm.LinearSVC(penalty='l1', loss="squared_hinge", C=1, dual=False)
    clf.fit(X_train, Y_train)
    p = performance(Y_test, clf.decision_function(X_test), metric='auroc')
    print(p)
    plot_weight(X_train, Y_train, 'l1', [.001,.01,.1,1], "squared_hinge", False)

    c_range = []
    dum = [[.01, .1, 1, 10, 100, 1000],[.01, .1, 1, 10, 100, 1000]]
    for c in dum[0]:
        for r in dum[1]:
            c_range.append([c, r])
    c_range.append(list(loguniform.rvs(a=.01, b=1000, size=25)))
    c_range.append(list(loguniform.rvs(a=.01, b=1000, size=25)))
    print(len(c_range))
    C, r = select_param_quadratic(X_train, Y_train, 5, 'auroc', np.array(c_range).T)
    print(C, r)

    clf = sklearn.svm.LinearSVC(penalty='l2', loss="hinge", C=.01, class_weight={-1: 40, 1: 10})
    cv_performance(clf, IMB_features, IMB_labels, k=5, metric="specificity")

    y_score = clf.decision_function(IMB_test_features)
    fpr, tpr, _ = metrics.roc_curve(IMB_test_labels, y_score)
    clf = sklearn.svm.LinearSVC(penalty='l2', loss="hinge", C=.01, class_weight={-1: 1, 1: 1})
    cv_performance(clf, IMB_features, IMB_labels, k=5, metric='specificity')
    y_score_imb = clf.decision_function(IMB_test_features)
    f,t,_ = metrics.roc_curve(IMB_test_labels, y_score_imb)
    plt.figure()
    lw=2
    plt.plot(
        fpr,
        tpr,
        color="red",
        lw=lw,
        label="ROC curve, Wn=40 Wp=10"
    )
    plt.plot(
        f,
        t,
        color="green",
        lw=lw,
        label="ROC curve, Wn=1, Wp=1"
    )
    plt.xlim([-.05,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.show()
    plot_weight(X_train, Y_train, "l2", [.001, .01, .1, 1, 10, 100, 1000], "hinge", True)
    avg = np.sum(X_train)/X_train.shape[0]
    print(avg)
    print(d)
    print(np.argmax(np.sum(xtr, axis=0)))
    Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # gratitude=1, neutral=0, sadness=-1
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()

    select_param_linear(multiclass_features, multiclass_labels, C_range=[.001, .01, .1, 1, 10, 100, 1000])
    clf = OneVsRestClassifier(LinearSVC(penalty='l2', loss="hinge", C=1,
                                        dual=True, random_state=445, class_weight='balanced'))
    clf.fit(multiclass_features, multiclass_labels)
    heldout_features = get_heldout_reviews(multiclass_dictionary)
    multi_pred = clf.predict(heldout_features)
    generate_challenge_labels(multi_pred, 'shagund')





if __name__ == "__main__":
    main()
