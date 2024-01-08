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

import base64
import requests

matplotlib.use('Qt5Agg')  # Backend image engine that works in pycharm
import matplotlib.pyplot as plt
import time

openai.api_key = os.getenv('OPENAI_API_KEY')

dataSetPath = '../../Dataset/'
captions_path = dataSetPath + 'annotations/captions_val2014.json'
inputs_path = '../my_inputs/'
results_path = '../my_results/'

## Load model
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()
context_length = model.context_length
vocab_size = model.vocab_size


## Sanity
def sanity():
    print('\n** Running Sanity **\n\n')

    # Load captions
    captions = loadCaptions(captions_path)

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
    captions = loadCaptions(captions_path)

    # Load images
    imagePath = dataSetPath + 'Semantics/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    # plotImages(my_images, imagePath)

    numImages = len(my_images['names'])

    # Load multiple choice file
    multChoiceFileName = 'semantics multiple choice.txt'
    multChoices = loadTextFile(os.path.join(inputs_path, multChoiceFileName))

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


def inContextLearning():
    print('\n** Running In-Context Learning **\n')





def main():
    print('Dataset path: ' + os.path.abspath(dataSetPath))
    print('Captions path: ' + os.path.abspath(captions_path))
    print('Inputs path: ' + os.path.abspath(inputs_path))

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # sanity()
    results = semantics()
    saveResultsToExcel(results, results_path, 'Semantics')



if __name__ == '__main__':
    # main()

    # Load captions
    captions = loadCaptions(captions_path)

    # Load images and find all captions per image
    imagePath = dataSetPath + 'Semantics/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    # plotImages(my_images, imagePath)


    # Start in-context learning
    # Describe mission
    numImages = 5
    myPrompt = """we are creating an image understanding test. 
                Given a few true image descriptions we want to generate wrong descriptions for a multiple-choice test. 
                The wrong captions need to not match the original image but to be close enough to be challenging and 
                test the reading comprehension."""

    printModelIo('Mission description: ' + myPrompt, True)

    # Generate a response from the model
    response = openai.Completion.create(
        # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
        model="gpt-3.5-turbo-instruct",
        # model="gpt-4-vision-preview",
        prompt=myPrompt,
        max_tokens=80,  # Adjust based on how long you expect the response to be
    )

    # Extract the text from the response
    printModelIo(response.choices[0].text.strip(), False)


    # Provide examples
    # Load examples file
    multChoiceFileName = 'In-context learning examples.txt'
    examples = loadTextFile(os.path.join(inputs_path, multChoiceFileName))

    myPrompt = '\n'.join(examples) + "\nPlease verify That you understood these examples."
    printModelIo(myPrompt, True)

    # Generate a response from the model
    response = openai.Completion.create(
        # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
        model="gpt-3.5-turbo-instruct",
        # model="gpt-4-vision-preview",
        prompt=myPrompt,
        max_tokens=80,  # Adjust based on how long you expect the response to be
    )
    # Extract the text from the response
    printModelIo(response.choices[0].text.strip(), False)

    # for i in range(numImages):
    #     imageId = my_images['imageIds'][i]
    #     trueCaptions = ""
    #     for j in range(len(my_images['captions'][imageId])):
    #         trueCaptions += my_images['captions'][imageId][j] + "\n"
    #
    #     Preface = "This is Example #" + str(i+1) + ". Following are 5 true sentences:"
    #     myPrompt = Preface + "\n" + trueCaptions + "Please verify That you understood this example."
    #     printModelIo(myPrompt, True)
    #
    #     # Generate a response from the model
    #     response = openai.Completion.create(
    #         # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
    #         model="gpt-3.5-turbo-instruct",
    #         # model="gpt-4-vision-preview",
    #         prompt=myPrompt,
    #         max_tokens=80,  # Adjust based on how long you expect the response to be
    #     )
    #     # Extract the text from the response
    #     printModelIo(response.choices[0].text.strip(), False)


    # Create alternative captions
    modelCaptions = {}
    for i in range(len(my_images['imageIds'])):
        imageId = my_images['imageIds'][i]
        trueCaptions = ""
        for j in range(len(my_images['captions'][imageId])):
            trueCaptions += my_images['captions'][imageId][j] + "\n"

        Preface = "This is Example #" + str(i+1) + ". Following are 5 true sentences:"
        task = ("Please create 5 similar yet wrong alternative captions. "
                "Please put each caption in a new line, without numbering the response.")
        myPrompt = Preface + "\n" + trueCaptions + task
        printModelIo(myPrompt, True)

        # Generate a response from the model
        response = openai.Completion.create(
            # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
            model="gpt-3.5-turbo-instruct",
            # model="gpt-4-vision-preview",
            prompt=myPrompt,
            max_tokens=100,  # Adjust based on how long you expect the response to be
        )
        # Extract the text from the response
        printModelIo(response.choices[0].text.strip(), False)

        # Keep results
        modelCaptions.update({imageId: response.choices[0].text.strip().split('\n')})


    # Create multiple choices text
    results = dict()
    for i in range(len(my_images['imageIds'])):
        imageId = my_images['imageIds'][i]
        if imageId not in list(modelCaptions.keys()):
            continue

        imageName = my_images['names'][i]
        choices = modelCaptions[imageId]
        choices.append(my_images['captions'][imageId][0])  # Correct sentence is last

        tempDict = cosineSimilarity(model, preprocess, choices, imagePath, imageName)
        results.update({imageName: tempDict})

    saveResultsToExcel(results, results_path, 'Semantics')
