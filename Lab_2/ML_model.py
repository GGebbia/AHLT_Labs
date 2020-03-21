import os
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer


def split_data(word_features):
    X_samples = []
    sample = []

    Y_labels = []
    label = []

    for word_feature in word_features:
        if word_feature != "":
            word_feature = word_feature.split(" ")
            label.append(word_feature[0])
            sample.append(word_feature[1:])
        else:
            X_samples.append(sample)
            Y_labels.append(label)

            sample = []
            label = []
    return X_samples, Y_labels


def output_predicted_entities(Y_pred, filename):
    Y_pred_flatten = [el for line in Y_pred for el in line]
    detailed_word_features = open(filename).read().split("\n")
    detailed_word_features = [el for el in detailed_word_features if el != ""]

    skip_next_word = False
    skip_next_word_times = 0
    for index, (label, word_feature) in enumerate(zip(Y_pred_flatten, detailed_word_features)):
        if skip_next_word:
            skip_next_word_times -= skip_next_word_times
            if skip_next_word_times == 0:
                skip_next_word = False
            continue

        if label == "O":
            continue
        else:
            label = label.split("-")[1]
            _id, word, offset_from, offset_to = word_feature.split(" ")[:4]

            # merge i-type with b-type
            next_word_index = 1
            while("I-" == Y_pred_flatten[index + next_word_index][0:2]):
                # todo ajuntar mes de un word
                _, new_word, _, new_offset_to = detailed_word_features[index+1].split(" ")[:4]
                offset_to = new_offset_to
                word = word + " " + new_word
                skip_next_word = True
                skip_next_word_times += skip_next_word_times
                next_word_index += 1

            line = [_id, "{}-{}".format(offset_from, offset_to), word, label]
            outputfile.write("|".join(line) + "\n")


def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateNER.jar " + inputdir + " " + outputfile)


### MAIN VARIABLES
train_filename = "data_Train_megam.dat"
# test_filename = "data_Devel_megam.dat"
test_filename = "data_Devel_megam.dat"

fulltest_filename = "data_Devel_feats.dat"
inputdir = "data/Devel"
output_filename = "predicted_Devel.txt"

train_samples = open(train_filename, "r").read().split("\n")
test_samples = open(test_filename, "r").read().split("\n")

X_train, Y_train = split_data(train_samples)
X_test, Y_test = split_data(test_samples)

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, Y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
   # 'c1': 1.0,  # coefficient for L1 penalty
    'c2': 0.1,  # coefficient for L2 penalty 1e-1 0.61
    'max_iterations': 250,  # stop earlier #todo put higher value

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train(train_filename + '_model.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open(train_filename + '_model.crfsuite')

Y_pred = [tagger.tag(xseq) for xseq in X_test]

outputfile = open(output_filename, "w")
output_predicted_entities(Y_pred, fulltest_filename)
outputfile.close()

evaluate(inputdir, output_filename)
