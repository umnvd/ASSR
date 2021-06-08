from __future__ import print_function, division, absolute_import
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms

ms.use('seaborn-muted')
# %matplotlib inline

import librosa
import soundfile as sf
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

# import tensorflow as tf
# from sklearn.model_selection import train_test_split


# Setting up progressbar and logger
progressbar.streams.wrap_stderr()
logger = colorlog.getLogger("ASSR")
handler = logging.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s| %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
        self.loge = None
        self.delta_loge = None
        self.delta2_loge = None
        self.wloge = None

    def loadFile(self, filename):
        self.y, self.sr = librosa.load(filename)
        logger.debug('File loaded: %s', filename)

    def load_y_sr(self, y, sr):
        self.y = y
        self.sr = sr

    def melspectrogram(self):
        self.S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=self.n_mels)
        self.log_S = librosa.core.power_to_db(self.S)

    def extractmfcc(self, n_mfcc=13):
        self.mfcc = librosa.feature.mfcc(S=self.log_S, n_mfcc=n_mfcc)
        self.delta_mfcc = librosa.feature.delta(self.mfcc, mode='constant')
        self.delta2_mfcc = librosa.feature.delta(self.mfcc, order=2, mode='constant')
        self.wmfcc = np.sum([self.mfcc,
                             self.delta_mfcc * self.delta_weight,
                             self.delta2_mfcc * self.delta2_weight], axis=0)

    def extractloge(self):
        energy = librosa.feature.melspectrogram(y=self.y, sr=self.sr, power=1)
        self.loge = librosa.core.amplitude_to_db(energy)
        self.delta_loge = librosa.feature.delta(self.loge, mode='constant')
        self.delta2_loge = librosa.feature.delta(self.loge, order=2, mode='constant')
        self.wloge = np.sum([self.loge,
                             self.delta_loge * self.delta_weight,
                             self.delta2_loge * self.delta2_weight], axis=0)


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
        num_lines = sum(1 for line in open(self.datasetLabelFilename, 'r'))
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
                    features.extractmfcc()
                    features.extractloge()
                except ValueError:
                    logger.warning("Error in extracting features from file %s", audiofilename)
                    continue

                featureVector = []
                for feature in features.wmfcc:
                    featureVector.append(np.mean(feature))

                featureVector.append(np.mean(features.wloge))

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

        splitDuration = 100  # milliseconds
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
                    sf.write(os.path.join(self.datasetDir, audiofilename), audio, sr)

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


# Entrypoint


def run(train=False, correct=False):
    if train:
        dataset = Dataset('dataset', 'datasetLabels.txt', 'datasetArray.gz')


# In[ ]:

if __name__ == "__main__":
    run(True, False)

