from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms

ms.use('seaborn-muted')
# %matplotlib inline

import librosa
import soundfile
import librosa.display
import IPython.display

import os
import sys
import re
import shutil
import datetime
import logging
import colorlog
import progressbar

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split


# Setting up progressbar and logger
progressbar.streams.wrap_stderr()
logger = colorlog.getLogger("ASSR")
handler = logging.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s| %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Weighted MFCC


class FeatureExtraction:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels
        self.delta_weight = 1 / 3  #
        self.delta2_weight = 1 / 6
        self.y = None
        self.sr = None
        self.S = None
        self.log_S = None
        self.mfcc = None
        self.delta_mfcc = None
        self.delta2_mfcc = None
        self.wmfcc = None
        self.rmse = None

    def loadFile(self, filename):
        self.y, self.sr = librosa.load(filename)
        logger.debug('File loaded: %s', filename)

    def load_y_sr(self, y, sr):
        self.y = y
        self.sr = sr

    def melspectrogram(self):
        self.S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=self.n_mels)
        self.log_S = librosa.core.power_to_db(self.S)

    def extractWMFCC(self, n_mfcc=13):
        self.mfcc = librosa.feature.mfcc(S=self.log_S, n_mfcc=n_mfcc)
        self.delta_mfcc = librosa.feature.delta(self.mfcc, mode='constant')
        self.delta2_mfcc = librosa.feature.delta(self.mfcc, order=2, mode='constant')
        self.wmfcc = np.sum([self.mfcc,
                             self.delta_mfcc * self.delta_weight,
                             self.delta2_mfcc * self.delta2_weight], axis=0)

    def extractRMSE(self):
        self.rmse = np.mean(librosa.feature.rms(self.y, self.S))


class Dataset:
    def __init__(self, datasetDir, datasetLabelFilename, datasetArrayFilename):
        self.n_features = 14
        logger.info("Number of features: %s", self.n_features)
        self.X = np.empty(shape=(0, self.n_features))
        self.Y = np.empty(shape=(0, 2))

        self.datasetArrayFilename = datasetArrayFilename
        logger.debug("Dataset array filename: %s", self.datasetArrayFilename)

        if os.path.isfile(self.datasetArrayFilename):
            self.__readFromFile()
        else:
            self.datasetDir = datasetDir
            logger.debug("Dataset Directory: %s", self.datasetDir)

            self.datasetLabelFilename = datasetLabelFilename
            logger.debug("Dataset labels filename: %s", self.datasetLabelFilename)

            if not os.path.isdir(self.datasetDir) or not os.path.isfile(self.datasetLabelFilename):
                logger.info("%s or %s does not exists", self.datasetDir, self.datasetLabelFilename)
                self.__buildDatasetAndLabels('records', 'recordsLabels.txt')

            self.__build()
            self.__writeToFile()

    def __build(self):
        logger.info("Building dataset from directory: %s", self.datasetDir)
        num_lines = sum(1 for _ in open(self.datasetLabelFilename, 'r'))
        with open(self.datasetLabelFilename, 'r') as datasetLabelFile:
            filesProcessed = 0
            pbar = progressbar.ProgressBar(redirect_stdout=True)
            for line in pbar(datasetLabelFile, max_value=num_lines):
                lineSplit = line.strip().split(' ')
                audiofilename = lineSplit[0]
                label = lineSplit[1]
                try:
                    features = FeatureExtraction()
                    features.loadFile(os.path.join(self.datasetDir, audiofilename))
                    features.melspectrogram()
                    features.extractWMFCC()
                    features.extractRMSE()
                except ValueError:
                    logger.warning("Error in extracting features from file %s", audiofilename)
                    continue

                featureVector = []
                for feature in features.wmfcc:
                    featureVector.append(np.mean(feature))

                featureVector.append(np.mean(features.rmse))

                self.X = np.vstack((self.X, [featureVector]))

                if label == "STUTTER":
                    self.Y = np.vstack((self.Y, [0, 1]))
                elif label == "NORMAL":
                    self.Y = np.vstack((self.Y, [1, 0]))
                else:
                    logger.error("Unexpected label: %s", label)
                    sys.exit()

                filesProcessed += 1

            logger.info("Total files processed: %d", filesProcessed)

    def __buildDatasetAndLabels(self, recordsDirectory, recordsLabelsFileName):
        logger.info("Rebuilding the dataset directory and labels")
        if os.path.isdir(self.datasetDir):
            shutil.rmtree(self.datasetDir)
        os.makedirs(self.datasetDir)

        labelFile = open(self.datasetLabelFilename, 'w')

        splitDuration = 300  # milliseconds
        pbar = progressbar.ProgressBar(redirect_stdout=True)
        with open(recordsLabelsFileName, 'r') as recordsLabelsFile:
            for line in pbar(recordsLabelsFile):

                logger.debug("Parsing line: %s", line)

                lineSplit = line.strip().split(';')
                subject = lineSplit[0]
                startTime = int(lineSplit[1])
                endTime = int(lineSplit[2])
                label = lineSplit[4]
                wavFileName = subject + '.wav'
                y, sr = librosa.load(os.path.join(recordsDirectory, wavFileName))

                n_splits = int(np.ceil((endTime - startTime) / splitDuration))

                for i in range(0, n_splits):
                    sampleStartTime = startTime + splitDuration * i
                    sampleEndTime = sampleStartTime + splitDuration
                    if sampleEndTime > endTime:
                        sampleEndTime = endTime
                    startingSample = int(sampleStartTime * sr / 1000)
                    endingSample = int(sampleEndTime * sr / 1000)
                    audiofilename = subject + "-" + str(sampleStartTime) + "-" + str(sampleEndTime) + ".wav"
                    labelFile.write(audiofilename + " " + label + "\n")
                    audio = y[startingSample:endingSample]
                    soundfile.write(os.path.join(self.datasetDir, audiofilename), audio, sr)

        labelFile.close()

    def __writeToFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename

        if os.path.exists(filename):
            os.remove(filename)
        np.savetxt(filename, np.hstack((self.X, self.Y)))
        logger.info("Array stored in file %s", filename)

    def __readFromFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename

        if not os.path.isfile(filename):
            logger.error("%s does not exists or is not a file", filename)
            sys.exit()
        matrix = np.loadtxt(filename)
        self.X = matrix[:, 0:self.n_features]
        self.Y = matrix[:, self.n_features:]
        logger.info("Array read from file %s", filename)


# Bi-LSTM classifier


class NeuralNetwork:
    def __init__(self, X_train=None, X_test=None, Y_train=None, Y_test=None):
        # Data
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Model configuration
        self.learning_rate = 0.001
        self.training_epochs = 1000
        self.batch_size = 100

        # Model
        self.model = self.__network()
        self.modelFile = None
        self.loss = None
        self.accuracy = None

    def __network(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(100), merge_mode='sum', input_shape=(1, 14)))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
        model.summary()

        return model

    def train(self):
        self.model.fit(self.X_train.reshape(-1, 1, 14), self.Y_train, batch_size=self.batch_size, epochs=self.training_epochs)
        self.__test()
        tfModelDir = "tfModels"
        if not os.path.isdir(tfModelDir):
            os.makedirs(tfModelDir)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        os.makedirs(os.path.join(tfModelDir, timestamp))
        self.modelFile = os.path.join(os.path.join(tfModelDir, timestamp), 'model.h5')
        self.model.save(self.modelFile)

        logger.info("Model saved in file: %s" % self.modelFile)

    def __test(self):
        # Test model
        test_results = self.model.evaluate(self.X_test.reshape(-1, 1, 14), self.Y_test)
        logger.info("Model tested")
        # Calculate accuracy
        self.loss = test_results[0]
        self.accuracy = 100 * test_results[1]
        logger.info("Loss: %f", self.loss)
        logger.info("Accuracy: %f", self.accuracy)

    def getModelFile(self):
        return self.modelFile

    def loadAndClassify(self, filename, X):
        self.model = tf.keras.models.load_model(filename)
        X = X.reshape(-1, 1, 14)
        return self.model.predict(X)


class AudioCorrection:
    def __init__(self, audiofile, modelFile, segmentLength=300, segmentHop=100, n_features=14,
                 correctionsDir='corrections'):
        self.modelFile = modelFile
        self.segmentLength = segmentLength
        self.segmentHop = segmentHop
        self.n_features = n_features
        self.correctionsDir = correctionsDir
        self.samplesPerSegment = None
        self.samplesToSkipPerHop = None
        self.upperLimit = None
        self.audioFile = audiofile
        self.y = None
        self.sr = None
        self.target_sr = 16000
        NORMAL = 0
        STUTTER = 1
        self.speech = {NORMAL: [], STUTTER: []}
        self.smoothingSamples = 1000
        self.__loadfile(audiofile)

    def __loadfile(self, inputFilename):
        if not os.path.isfile(inputFilename):
            logger.error("%s does not exists or is not a file", inputFilename)
            sys.exit()
        self.audioFile = inputFilename
        logger.info("Loading file %s", self.audioFile)
        self.y, self.sr = librosa.load(self.audioFile)
        self.samplesPerSegment = int(self.segmentLength * self.sr / 1000)
        self.samplesToSkipPerHop = int(self.segmentHop * self.sr / 1000)
        self.upperLimit = len(self.y) - self.samplesPerSegment

    def process(self):
        logger.info("Attempting to correct %s", self.audioFile)
        X = np.empty(shape=(0, self.n_features))
        durations = np.empty(shape=(0, 2))

        pbar = progressbar.ProgressBar()
        start = 0
        end = 0
        for start in pbar(range(0, self.upperLimit, self.samplesToSkipPerHop)):
            end = start + self.samplesPerSegment
            audio = self.y[start:end]

            featureVector = self.__getFeatureVector(audio, self.sr)
            if featureVector != None:
                X = np.vstack((X, [featureVector]))
                durations = np.vstack((durations, [start, end]))

        audio = self.y[end:]
        featureVector = self.__getFeatureVector(audio, self.sr)
        if featureVector != None:
            X = np.vstack((X, [featureVector]))
            durations = np.vstack((durations, [end, self.upperLimit + self.samplesPerSegment]))
        logger.debug("Finished extracting features")

        nn = NeuralNetwork()
        classificationResult = nn.loadAndClassify(self.modelFile, X)
        classificationResult = np.concatenate(np.hsplit(classificationResult, 2)[1])
        classificationResult = np.round(classificationResult).astype(int)
        logger.debug("Finished classification of segments")

        currentSegment = {'type': classificationResult[0], 'start': durations[0][0], 'end': durations[0][1]}
        for (label, [start, end]) in zip(classificationResult[1:], durations[1:]):
            if currentSegment['type'] == label:
                currentSegment['end'] = end
            else:
                self.speech[currentSegment['type']].append((currentSegment['start'], currentSegment['end']))
                currentSegment['type'] = label
                currentSegment['start'] = start
                currentSegment['end'] = end

    def __getFeatureVector(self, y, sr):
        try:
            features = FeatureExtraction()
            features.load_y_sr(y, sr)
            features.melspectrogram()
            features.extractWMFCC()
            features.extractRMSE()
        except ValueError:
            logger.warning("Error extracting features")
            return None

        featureVector = []
        for feature in features.wmfcc:
            featureVector.append(np.mean(feature))

        featureVector.append(np.mean(features.rmse))
        return featureVector

    def saveCorrectedAudio(self):
        NORMAL = 0
        STUTTER = 1
        if not os.path.isdir(self.correctionsDir):
            os.makedirs(self.correctionsDir)
        outputFilenamePrefix = os.path.join(self.correctionsDir,
                                            os.path.splitext(os.path.basename(self.audioFile))[0])

        normalSpeech = np.ndarray(shape=(1, 0))
        (start, end) = self.speech[NORMAL][0]
        normalSpeech = np.append(normalSpeech, self.y[int(start):int(end)])
        for (start, end) in self.speech[NORMAL][1:]:
            # Smoothing
            previousSample = normalSpeech[-1]
            nextSample = self.y[int(start)]
            if nextSample > previousSample:
                low, high = previousSample, nextSample
            else:
                low, high = nextSample, previousSample

            step = (high - low) / self.smoothingSamples

            try:
                normalSpeech = np.append(normalSpeech, np.arange(low, high, step))
                normalSpeech = np.append(normalSpeech, self.y[int(start):int(end)])
            except Exception as e:
                print(e)

        stutteredSpeech = np.ndarray(shape=(1, 0))
        for (start, end) in self.speech[STUTTER]:
            stutteredSpeech = np.append(stutteredSpeech, self.y[int(start):int(end)])

        # Resampling the audio
        logger.debug("Resampling corrected audio from %d to %d", self.sr, self.target_sr)
        # resampledNormalSpeech = librosa.resample(normalSpeech, self.sr, self.target_sr)
        # resampledStutteredSpeech = librosa.resample(stutteredSpeech, self.sr, self.target_sr)
        soundfile.write(outputFilenamePrefix + "-corrected.wav", normalSpeech, self.sr)
        # soundfile.write(outputFilenamePrefix + "-stuttered.wav", stutteredSpeech, self.sr)
        logger.info("Corrected audio saved as %s", outputFilenamePrefix + "-corrected.wav")


def run(train=False, correct=False):
    nn = None
    if train:
        dataset = Dataset('dataset', 'datasetLabels.txt', 'datasetArray.gz')
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y, train_size=0.8)

        print(np.shape(X_train))
        print(np.shape(Y_train))

        nn = NeuralNetwork(X_train, X_test, Y_train, Y_test)
        nn.train()

    if correct:
        audiofile = 'records/s3_p3.wav'
        if train:
            modelFile = nn.getModelFile()
        else:
            modelFile = 'tfModels/2021-06-16-05.56.37/model.h5'

        correction = AudioCorrection(audiofile, modelFile)
        correction.process()
        correction.saveCorrectedAudio()


if __name__ == "__main__":
    run(True, True)

