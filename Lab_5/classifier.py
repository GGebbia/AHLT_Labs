#!/usr/bin/env python

def predict(modelname, datadir, outfile):
    '''
    Loads a NN model from file ’modelname ’ and uses it to extract drugs
    in datadir . Saves results to ’outfile ’ in the appropriate format .
    '''

    # load model and associated encoding data
    model, idx = load_model_and_indexs(modelname)
    # load data to annotate
    testdata = load_data(datadir)

    # encode dataset
    X = encode_words(testdata, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each pair
    Y = [[idx['tags'][np.argmax(y)] for y in Y]

    # extract entities and dump them to output file
    output_interactions(testdata, Y, outfile)

    # evaluate using official evaluator
    evaluation(datadir, outfile)
