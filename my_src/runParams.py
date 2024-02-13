from globals import *
import datetime
import os

class RunParams:

    def __init__(self, runName, numImages):

        self.dataSetPath = dataSetPath
        self.captionsPath = captionsPath
        self.imagePath = imagePath
        self.inputsPath = inputsPath
        self.tempPath = tempPath
        self.resultsPath = resultsPath
        self.encodedPath = encodedPath
        self.numImages = numImages

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runName = runName + '_' + timestamp
        self.promptFileTemp = runName + '_prompt.log'

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
        dataSet path: {dataSetPath}
        captions path = {captionsPath}
        image path = {imagePath}
        inputs path = {inputsPath}
        results path = {resultsPath}
        """
