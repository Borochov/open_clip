import os
import math
import numpy as np
import pandas as pd
import torch
import open_clip
from open_clip import tokenizer
import openai
from PIL import Image
from my_utils import *
import matplotlib
matplotlib.use('Qt5Agg')  # Backend image engine that works in pycharm
import matplotlib.pyplot as plt
import time
import base64
import requests

openai.api_key = os.getenv('OPENAI_API_KEY')

dataSetPath = '../../Dataset/'
captionsPath = dataSetPath + 'annotations/captions_val2014.json'
inputsPath = '../my_inputs/'
resultsPath = '../my_results/'

## Load model
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()
context_length = model.context_length
vocab_size = model.vocab_size


## Sanity
def sanity():
    print('\n** Running Sanity **\n\n')

    # Load captions
    captions = loadCaptions(captionsPath)

    # Load images
    imagePath = dataSetPath + 'Sanity/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    plotImages(my_images, imagePath)

    ## Build features
    images, texts, original_images = [], [], []
    numImages = len(my_images['names'])
    for i in range(numImages):
        image = Image.open(os.path.join(imagePath, my_images['names'][i])).convert("RGB")
        original_images.append(image)
        images.append(preprocess(image))
        texts.append(my_images['captions'][i])

    image_input = torch.tensor(np.stack(images))
    text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

    print(f"Started model encoding.")
    # Timer
    start_time = time.time()
    with torch.no_grad():
        image_features = model.encodeImageBase64(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    # Timer
    end_time = time.time()
    duration = end_time - start_time
    print(f"Model encoding took {round(duration)} seconds to run.")

    ## Calculate cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    ## Display cosine similarity
    plotSimilarity(similarity, original_images, texts, numImages)

    return


## Semantics
def semantics():

    print('\n** Running Semantics **\n')

    # Load captions
    captions = loadCaptions(captionsPath)

    # Load images
    imagePath = dataSetPath + 'Semantics/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    # plotImages(my_images, imagePath)

    numImages = len(my_images['names'])

    # Load multiple choice file
    multChoiceFileName = 'semantics multiple choice.txt'
    multChoices = loadTextFile(os.path.join(inputsPath, multChoiceFileName))

    results = dict()
    for i in range(numImages):

        # Find multiple choices for this image
        key = list(my_images['captions'].keys())[i]
        caption = my_images['captions'][key][0]
        choices = []
        try:
            multIdx = multChoices.index(caption)
            choices = multChoices[multIdx:multIdx+4]

        except ValueError:
            print('Could not find multiple choices for: ' + caption)
            continue

        tempDict = cosineSimilarity(model, preprocess, choices, imagePath, my_images['names'][i])
        results.update({my_images['names'][i]: tempDict})

    return results


def uploadLocalImage():
    # Path to your image
    imagePath = "C:/Work/Python/Thesis/Dataset/Semantics/COCO_val2014_000000005577.jpg"

    # Getting the base64 string
    base64Image = encodeImageBase64(imagePath)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64Image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 60
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())


def inContextLearning(my_images):
    print('\n** Running In-Context Learning **\n')

    modelCaptions = {}
    for imageIdx in range(len(my_images['imageIds'])):

        # Describe mission
        myPrompt = """This is a new task. We are creating an image understanding test. 
                    Given a few true image descriptions we want to generate wrong descriptions for a multiple-choice test. 
                    The wrong captions need to not match the original image but to be close enough to be challenging and 
                    test the reading comprehension."""

        printModelIo('Mission description: ' + myPrompt, True)

        # Get response from the model
        response = getModelResponse(openai, myPrompt)
        # printModelIo(response, False)


        # Provide examples
        # Load examples file
        multChoiceFileName = 'In-context learning examples - Semantics 6.txt'
        examples = loadTextFile(os.path.join(inputsPath, multChoiceFileName))

        myPrompt = '\n'.join(examples) + "\nPlease verify That you understood these examples."
        printModelIo(myPrompt, True)

        # Get response from the model
        response = getModelResponse(openai, myPrompt)
        printModelIo(response, False)


        # Create alternative captions
        imageId = my_images['imageIds'][imageIdx]
        trueCaptions = ""
        for j in range(len(my_images['captions'][imageId])):
            trueCaptions += my_images['captions'][imageId][j] + "\n"

        Preface = "Following are 5 true sentences:"
        task = ("Please create 5 similar yet wrong alternative captions. "
                "Please put each caption in a new line, without numbering the response.")
        myPrompt = Preface + "\n" + trueCaptions + task
        printModelIo(myPrompt, True)

        # Get response from the model
        response = getModelResponse(openai, myPrompt)
        printModelIo(response, False)

        # Keep results
        modelCaptions.update({imageId: response.split('\n')})

    return modelCaptions

## Test model using multiple choices
def testModel(imagePath, my_images, modelCaptions):
    # Create multiple choices text
    results = dict()
    for i in range(len(my_images['imageIds'])):
        imageId = my_images['imageIds'][i]
        if imageId not in list(modelCaptions.keys()):
            continue

        imageName = my_images['names'][i]
        choices = modelCaptions[imageId]
        choices.append(my_images['captions'][imageId][0])  # Correct sentence is last

        # Run cosine similarity test
        tempDict = cosineSimilarity(model, preprocess, choices, imagePath, imageName)
        results.update({imageName: tempDict})

    return results



def main(runName):
    print('Dataset path: ' + os.path.abspath(dataSetPath))
    print('Captions path: ' + os.path.abspath(captionsPath))
    print('Inputs path: ' + os.path.abspath(inputsPath))

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # sanity()
    # results = semantics()

    # Load captions
    captions = loadCaptions(captionsPath)

    # Load images and find all captions per image
    imagePath = dataSetPath + 'Semantics6/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    # plotImages(my_images, imagePath)

    modelCaptions = inContextLearning(my_images)

    print('Running Model...')
    results = testModel(imagePath, my_images, modelCaptions)
    # saveResultsToExcel(results, resultsPath, 'Semantics')
    saveImagesWithCaptions(my_images, results, imagePath, resultsPath, runName)


if __name__ == '__main__':
    runName = 'Semantics6'
    main(runName)
