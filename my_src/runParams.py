from globals import *
import datetime
import time
import os
import shutil


class RunParams:

    def __init__(self, runName, numImages):

        self.dataSetPath = dataSetPath
        self.captionsPath = captionsPath
        self.imagePath = imagePath
        self.inputsPath = inputsPath
        self.missionPath = os.path.join(inputsPath, missionFileName)
        self.examplesPath = os.path.join(inputsPath, examplesFileName)
        self.taskPath = os.path.join(inputsPath, taskFileName)
        self.tempPath = tempPath
        self.resultsPath = resultsPath
        self.encodedPath = encodedPath
        self.numImages = numImages
        self.start_time = time.time()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runName = runName + '_' + timestamp
        self.promptFileTemp = runName + '_prompt.log'

        self.gptModel = GPT_MODEL

        # Create run directories
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)

        # Create results dir
        self.resultsDir = os.path.join(self.resultsPath + self.runName + '/')
        if not os.path.exists(self.resultsDir):
            # Create results directory if it does not exist
            os.makedirs(self.resultsDir)
            os.makedirs(self.resultsDir + 'right')
            os.makedirs(self.resultsDir + 'wrong')
            print(f'Created results directory: ' + self.resultsDir)


    def __str__(self):
        return f"""
        *** {self.runName} ***
        --------------------\n
        start time: {datetime.datetime.fromtimestamp(self.start_time).strftime("%d/%m/%Y %H:%M:%S")}
        number of images: {self.numImages}
        GPT model: {self.gptModel}
        dataSet path: {self.dataSetPath}
        captions path: {self.captionsPath}
        image path: {self.imagePath}
        encoded images path: {self.encodedPath}
        inputs path: {self.inputsPath}
        mission file: {self.missionPath}
        examples file: {self.examplesPath}
        task file: {self.taskPath}
        results dir: {self.resultsDir}
        """

    def save(self):
        try:
            with open(os.path.join(self.resultsDir, 'runParams.txt'), 'a') as file:
                file.write(str(self) + "\n")
        except IOError as e:
            print(f"Unable to write to file: {e}")
