from __future__ import absolute_import	
from __future__ import division	
from __future__ import print_function	
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam

import logging

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE

import numpy as np
import pickle
import time

dpsgd = False               # If True, train with DP-SGD
noise_multiplier = 0.3      # Ratio of the standard deviation to the clipping norm
l2_norm_clip = 0.5          # Clipping norm
flow_size = 100
BATCH_SIZE = 32             # 4
microbatches = 1            # new
learn_rate = 0.0001         # tem de ser assim!
learn_rate_dp = 0.0015
negative_samples=1
NUM_PREDICTIONS = 5
TRAIN = False

reduce_factor = 4
orig = [2048/reduce_factor, 1024/reduce_factor, 32768/reduce_factor, 4096/reduce_factor, 512/reduce_factor, 128/reduce_factor]
dropout_prob = 0

print('Loading configurations')
config = tfe.RemoteConfig.load('config_local.json')
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

TRAINCONFIG=input('\nDo you want to train the model? ')

if TRAINCONFIG=='y':
    TRAIN = True

DIFFERENTIALPRIVACY=input('\nDifferential privacy? ')

if DIFFERENTIALPRIVACY=='y':
    dpsgd = True

datasetScalerTimes = 1000
datasetScalerSizes = 1/1000

CNN_21_41 = {'kernelSize1':[2,2], 'stride1':[2,2], 'kernelSize2':[4,4], 'stride2':[4,4], 'poolSize1': [1,5], 'poolStride1': [1,1], 'poolSize2': [1,5], 'poolStride2': [1,1]} 

baseDatasetPath = '/home/djfa29/cnns_dataset/'
# baseDatasetPath = '/media/sf_Datasets/cnns_dataset/'

metaDataPairsFolders = pickle.load(open(baseDatasetPath + 'metaDataPairsFolders', 'rb'))
trainPairsFolders = pickle.load(open(baseDatasetPath + 'trainPairsFolders', 'rb'))
testPairsFolders = pickle.load(open(baseDatasetPath + 'testPairsFolders', 'rb'))

onionAddressData = metaDataPairsFolders['onionAddressData']


def compute_epsilon(steps):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = BATCH_SIZE / 3200
    rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                    orders=orders)
    # Delta is set to 1e-4 because this dataset has 3200 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-4)[0]


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
        for negetive_pair_index in random:

            #skip if is the original correlated pair
            if pairsFoldersInput[negetive_pair_index]['hsFolder'] == pairFolder['hsFolder']:
                continue

            hsTimesIn = pairsFoldersInput[negetive_pair_index]['hsFlow']['timesIn']
            hsTimesOut = pairsFoldersInput[negetive_pair_index]['hsFlow']['timesOut']
            hsSizesIn = pairsFoldersInput[negetive_pair_index]['hsFlow']['sizesIn']
            hsSizesOut = pairsFoldersInput[negetive_pair_index]['hsFlow']['sizesOut']

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

        tf.logging.set_verbosity(tf.logging.INFO)
        if dpsgd and BATCH_SIZE % microbatches != 0:
            raise ValueError('Number of microbatches should divide evenly batch_size')

    return np.asarray(l2s), np.asarray(labels)

l2s_train, labels_train = generateDataset(trainPairsFolders)
l2s_train = l2s_train[:-26]
labels_train = labels_train[:-26]
l2s_test, labels_test = generateDataset(testPairsFolders)
l2s_test = l2s_test[:-8]
labels_test = labels_test[:-8]


def load_model():

    EPOCH_COUNT = 15

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(int(orig[0]), batch_input_shape=[BATCH_SIZE, 8, flow_size, 1], kernel_size=CNN_21_41['kernelSize1'], strides=CNN_21_41['stride1'], padding='VALID', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=CNN_21_41['poolSize1'], strides=CNN_21_41['poolStride1'], padding='VALID'))

    model.add(keras.layers.Conv2D(int(orig[1]), kernel_size=CNN_21_41['kernelSize2'], strides=CNN_21_41['stride2'], padding='VALID', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=CNN_21_41['poolSize2'], strides=CNN_21_41['poolStride2'], padding='VALID'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(int(orig[3]), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(int(orig[4]), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(int(orig[5]), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))


    def customLoss(y_true,y_pred):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y_true),name='loss_sigmoid')

    def customLoss_2(y_true,y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y_true)

    if dpsgd:
        # optimizer = DPAdamGaussianOptimizer(
        #     l2_norm_clip=l2_norm_clip,
        #     noise_multiplier=noise_multiplier,
        #     num_microbatches=microbatches,
        #     learning_rate=learn_rate)
        
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=microbatches,
            learning_rate=learn_rate_dp,
            gradient_accumulation_steps=2)

        # Compute loss as a tensor. Do not call tf.reduce_mean as you would with a standard optimizer.

        loss = customLoss_2

        checkpoint_path = "checkpoints/correlation/checkpoints_privacy_batch_size=" + str(BATCH_SIZE) + "/"

    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        loss = customLoss
        checkpoint_path = "checkpoints/correlation/checkpoints_server_batch_size=" + str(BATCH_SIZE) + "/"


    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    if TRAIN:

        start = time.time()

        history = model.fit(l2s_train, labels_train, epochs=EPOCH_COUNT, validation_data=(l2s_test, labels_test), batch_size=BATCH_SIZE, callbacks=[cp_callback])
        
        end = time.time()
        print(f'Elapsed time: {str(end - start)} seconds')

        # Loss plot
        loss_plot = plt.figure() 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss over the Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')

        if dpsgd:
            plot_name = 'plots/correlation/DP/loss_plot_batch_size_' + str(BATCH_SIZE) + '.png'
        else:
            plot_name = 'plots/correlation/without_DP/loss_plot_batch_size_' + str(BATCH_SIZE) + '.png'

        plt.savefig(plot_name)

        # Accuracy plot
        accuracy_plot = plt.figure() 
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy over the Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')

        if dpsgd:
            plot_name_2 = 'plots/correlation/DP/accuracy_plot_batch_size_' + str(BATCH_SIZE) + '.png'
        else:
            plot_name_2 = 'plots/correlation/without_DP/accuracy_plot_batch_size_' + str(BATCH_SIZE) + '.png'
            
        plt.savefig(plot_name_2)

        # Compute epsilon

        if dpsgd:
            eps = compute_epsilon(EPOCH_COUNT * 3200 // BATCH_SIZE)
            print('For delta=1e-4, the current epsilon is: %.2f' % eps)
        else:
            print('Trained with vanilla non-private Adam optimizer')
    
    else:
        
        model.load_weights(checkpoint_path) 

    loss, acc = model.evaluate(l2s_test, labels_test, verbose=2)
    print("\nRestored model, accuracy: {:5.2f}%\n".format(100 * acc))

    tfe_model = tfe.keras.models.clone_model(model)

    return tfe_model


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    tfe_model = load_model()

    with tfe.protocol.SecureNN():

        q_input_shape = (1, 8, flow_size, 1)
        q_output_shape = (1, 1)

        server = tfe.serving.QueueServer(input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model)

        sess = KE.get_session()

        request_ix = 1

        def step_fn():
            global request_ix
            print("Served encrypted prediction {i} to client.".format(i=request_ix))
            request_ix += 1

        start = time.time()

        server.run(sess, num_steps=NUM_PREDICTIONS, step_fn=step_fn)

        end = time.time()
        print(f'Elapsed time: {str(end - start)} seconds')