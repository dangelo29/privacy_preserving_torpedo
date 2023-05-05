import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
import pickle
import math

baseDatasetPath = '/home/djfa29/cnns_dataset/'

metaDataPairsFolders = pickle.load(open(baseDatasetPath + 'metaDataPairsFolders', 'rb'))
trainPairsFolders = pickle.load(open(baseDatasetPath + 'trainPairsFolders', 'rb'))
testPairsFolders = pickle.load(open(baseDatasetPath + 'testPairsFolders', 'rb'))

onionAddressData = metaDataPairsFolders['onionAddressData']

datasetScalerTimes = 1000
datasetScalerSizes = 1/1000

negative_samples=1

flow_size = 100
threshold = 0.75            # usually good when 0.75 <= threshold <= 0.8 for ranking stage and 0.6 <= threshold <= 0.75 for correlation stage

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def round(x):
    if x >= threshold:
        return 1
    else:
        return 0

def generateDataset(pairsFoldersInput):

    global negative_samples, flow_size, onionAddressData, graphTitle

    for onionUrl in onionAddressData:
        onionAddressData[onionUrl]['connectionIndex'] = 0

    index = 0

    l2s=np.zeros((len(pairsFoldersInput)*(negative_samples+1),8,flow_size,1))
    labels=np.zeros((len(pairsFoldersInput)*(negative_samples+1),1))
    for pairFolder in pairsFoldersInput:
        
        clientTimesIn = pairFolder['clientFlow']['timesIn']
        clientTimesOut = pairFolder['clientFlow']['timesOut']
        clientSizesIn = pairFolder['clientFlow']['sizesIn']
        clientSizesOut = pairFolder['clientFlow']['sizesOut']

        hsTimesIn = pairFolder['hsFlow']['timesIn']
        hsTimesOut = pairFolder['hsFlow']['timesOut']
        hsSizesIn = pairFolder['hsFlow']['sizesIn']
        hsSizesOut = pairFolder['hsFlow']['sizesOut']

        l2s[index,0,:len(clientTimesIn[:flow_size]),0]=np.array(clientTimesIn[:flow_size])*datasetScalerTimes
        l2s[index,1,:len(hsTimesOut[:flow_size]),0]=np.array(hsTimesOut[:flow_size])*datasetScalerTimes
        l2s[index,2,:len(hsTimesIn[:flow_size]),0]=np.array(hsTimesIn[:flow_size])*datasetScalerTimes
        l2s[index,3,:len(clientTimesOut[:flow_size]),0]=np.array(clientTimesOut[:flow_size])*datasetScalerTimes

        l2s[index,4,:len(clientSizesIn[:flow_size]),0]=np.array(clientSizesIn[:flow_size])*datasetScalerSizes
        l2s[index,5,:len(hsSizesOut[:flow_size]),0]=np.array(hsSizesOut[:flow_size])*datasetScalerSizes
        l2s[index,6,:len(hsSizesIn[:flow_size]),0]=np.array(hsSizesIn[:flow_size])*datasetScalerSizes
        l2s[index,7,:len(clientSizesOut[:flow_size]),0]=np.array(clientSizesOut[:flow_size])*datasetScalerSizes
    
        labels[index, 0] = 1

        index += 1

        random = list(range(len(pairsFoldersInput)))
        np.random.shuffle(random)
        negative_samples_current = 0
        for negative_pair_index in random:

            #skip if is the original correlated pair
            if pairsFoldersInput[negative_pair_index]['hsFolder'] == pairFolder['hsFolder']:
                continue

            hsTimesIn = pairsFoldersInput[negative_pair_index]['hsFlow']['timesIn']
            hsTimesOut = pairsFoldersInput[negative_pair_index]['hsFlow']['timesOut']
            hsSizesIn = pairsFoldersInput[negative_pair_index]['hsFlow']['sizesIn']
            hsSizesOut = pairsFoldersInput[negative_pair_index]['hsFlow']['sizesOut']

            l2s[index,0,:len(clientTimesIn[:flow_size]),0]=np.array(clientTimesIn[:flow_size])*datasetScalerTimes
            l2s[index,1,:len(hsTimesOut[:flow_size]),0]=np.array(hsTimesOut[:flow_size])*datasetScalerTimes
            l2s[index,2,:len(hsTimesIn[:flow_size]),0]=np.array(hsTimesIn[:flow_size])*datasetScalerTimes
            l2s[index,3,:len(clientTimesOut[:flow_size]),0]=np.array(clientTimesOut[:flow_size])*datasetScalerTimes

            l2s[index,4,:len(clientSizesIn[:flow_size]),0]=np.array(clientSizesIn[:flow_size])*datasetScalerSizes
            l2s[index,5,:len(hsSizesOut[:flow_size]),0]=np.array(hsSizesOut[:flow_size])*datasetScalerSizes
            l2s[index,6,:len(hsSizesIn[:flow_size]),0]=np.array(hsSizesIn[:flow_size])*datasetScalerSizes
            l2s[index,7,:len(clientSizesOut[:flow_size]),0]=np.array(clientSizesOut[:flow_size])*datasetScalerSizes

            labels[index, 0] = 0

            index += 1

            negative_samples_current += 1
            if negative_samples_current >= negative_samples:
                break

        onionAddressData[pairFolder['onionAddress']]['connectionIndex'] += 1

    return np.asarray(l2s), np.asarray(labels)

# truncate the dataset so it divides evenly by the batch size (maximum is 32)
l2s_train, labels_train = generateDataset(trainPairsFolders)
l2s_train = l2s_train[:-26]
labels_train = labels_train[:-26]
l2s_test, labels_test = generateDataset(testPairsFolders)
l2s_test = l2s_test[:-8]
labels_test = labels_test[:-8]

# reshape datasets into the input shape of the server queue 
l2s_train = l2s_train.reshape(labels_train.shape[0], 8, 100, 1)
l2s_test = l2s_test.reshape(labels_test.shape[0], 8, 100, 1)

# load configurations and define MPC protocol 
config = tfe.RemoteConfig.load("config.json")
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

q_input_shape = (1, 8, flow_size, 1)
q_output_shape = (1, 1)

client = tfe.serving.QueueClient(input_shape=q_input_shape, output_shape=q_output_shape)

sess = tfe.Session(config=config)

# number of inferences (5 seems okay for correlation, but can be 10 or higher for ranking)
NUM_PREDICTIONS = 5

predictions, expected_labels = l2s_test[:NUM_PREDICTIONS], labels_test[:NUM_PREDICTIONS]
for prediction, expected_label in zip(predictions, expected_labels):
    
    res = client.run(
        sess,
        prediction.reshape(1, 8, flow_size, 1))
    
    predicted_label = sigmoid(res[0][0])
    predicted_label_round = round(predicted_label)
    
    print("The flow pair had label {} and was {} classified as {} (actual value = {})".format(
        int(expected_label[0]),
        "correctly" if int(expected_label[0]) == predicted_label_round else "incorrectly",
        predicted_label_round, predicted_label))