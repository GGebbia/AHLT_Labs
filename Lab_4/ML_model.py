import os
import pycrfsuite
import argparse
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


def split_data(features):
    """
    Receive a list of featured sentences splitted by words and split it into samples and labels.

    Parameters:
    word_features(list): List of feature words. There is an empty element between sentences in order to split each one.

    Returns:
    X_samples(list): List of feature words missing the label of the word.
    Y_labels(list): List of labels for each word.
    """

    X_samples = []
    Y_labels = []

    for feat in features:
        feat = feat.split(" ")
        Y_labels.append([feat[0]])
        X_samples.append([feat[1:]])

    return X_samples, Y_labels


def output_predicted_entities(Y_pred, filename):
    """
    Receives the predicted list of labels by the ML model and a filename of the detailed word features and construct predicted.txt file
    with the intrinsic details of each word, id, offsets and the predicted label.

    Parameters:
    Y_pred(list): List of list of predicted labels for each word.
    filename(str): filename to read the "feats.dat" data of the testing dataset

    Returns:

    """

    Y_pred_flatten = [el for line in Y_pred for el in line]
    detailed_word_features = open(filename).read().split("\n")[:-1]

    for label, detailed_feats in zip(Y_pred_flatten, detailed_word_features):
        _id, e1_id, e2_id = detailed_feats.split(" ")[0:3]
        if label == "null":
            interaction = "0"
        else:
            interaction = "1"
        line = [_id, e1_id, e2_id, interaction, label]
        outputfile.write("|".join(line) + "\n")
        
def evaluate(inputdir, outputfile):
    """
    Receives an input directory and the outputfile to evaluate the predicted labels with the evaluateNER.jar program.

    Parameters:
    inputdir(str):
    outputfile(str):

    Returns:

    """

    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)



parser = argparse.ArgumentParser(description=
                                 """
        Compute the Machine Learning model loading the features extracted on the training dataset and applying on the devel/testing dataset.
        """
                                 )
parser.add_argument('--train', type=str, help='Path to Train dataset directory to use.')
parser.add_argument('--test', type=str, help='Path to Test dataset directory to use')
parser.add_argument('--out', type=str, help='Output directory to save the results.')
args = parser.parse_args()


### MAIN VARIABLES
train_filename = os.path.join(args.train, "megam.dat")
test_filename = os.path.join(args.test, "megam.dat")
fulltest_filename = os.path.join(args.test, "feats.dat")

# _type represents the if it has internal or external knowledge
_type = args.test.split("/")[-2]
inputdir = os.path.join("data", args.test.split("/")[-1])

# If outputdir does not exist, we safely create it
outputdir = args.out
if not os.path.exists(outputdir):
    os.makedirs(outputdir)


output_filename = os.path.join(outputdir, "predicted.txt")
model_filename = os.path.join(outputdir, "model.crfsuite")

train_samples = open(train_filename, "r").read().split("\n")[:-1]
test_samples = open(test_filename, "r").read().split("\n")[:-1]

X_train, Y_train = split_data(train_samples)
X_test, Y_test = split_data(test_samples)

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, Y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.05,  # coefficient for L1 penalty
    'c2': 0.1,  # coefficient for L2 penalty 1e-1 0.61
    'max_iterations': 10000,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train(model_filename)

tagger = pycrfsuite.Tagger()
tagger.open(model_filename)

Y_pred = [tagger.tag(xseq) for xseq in X_test]

outputfile = open(output_filename, "w")
output_predicted_entities(Y_pred, fulltest_filename)
outputfile.close()

evaluate(inputdir, output_filename)