import os
import math
import numpy as np
import pandas as pd
import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from my_utils import *
import matplotlib

matplotlib.use('Qt5Agg')  # Backend image engine that works in pycharm
import matplotlib.pyplot as plt
import time

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
        image_features = model.encode_image(image_input).float()
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

    print('\n** Running Semantics **\n\n')

    # Load captions
    captions = loadCaptions(captions_path)

    # Load images
    imagePath = dataSetPath + 'Semantics3/'
    my_images = findCaptionForImage(imagePath, captions)

    # Plot images and matching captions
    # plotImages(my_images, imagePath)

    numImages = len(my_images['names'])
    # numImages = 10

    # Load multiple choice file
    multChoiceFileName = 'semantics multiple choice 3.txt'
    multChoices = loadTextFile(os.path.join(inputs_path, multChoiceFileName))

    results = dict()
    for i in range(numImages):

        # Create 3 negative sentences
        # text = "This is " + my_images['captions'][i]
        # newText = negateCaptions(my_images['captions'][i])

        # Find multiple choices for this image
        caption = my_images['captions'][i]
        choices = []
        try:
            multIdx = multChoices.index(caption)
            choices = multChoices[multIdx:multIdx+4]

        except ValueError:
            print('Could not find multiple choices for: ' + caption)
            continue

        text_tokens = tokenizer.tokenize(["This is " + desc for desc in choices])
        image = Image.open(os.path.join(imagePath, my_images['names'][i])).convert("RGB")
        image_input = torch.tensor(np.stack([preprocess(image)]))

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        ## Calculate cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        ## Display cosine similarity
        # plotSimilarity(similarity, [image], choices, 4)

        tempDict = dict()
        tempDict.update({'choices': choices})
        tempDict.update({'similarity': similarity.squeeze().tolist()})
        results.update({my_images['names'][i]: tempDict})

    return results


def main():
    print('Dataset path: ' + os.path.abspath(dataSetPath))
    print('Captions path: ' + os.path.abspath(captions_path))
    print('Inputs path: ' + os.path.abspath(inputs_path))

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # sanity()
    results = semantics()

    for k in list(results.keys()):
        # Convert to a DataFrame
        x = np.array(results[k]['choices'])
        y = np.array(results[k]['similarity'])
        df = pd.DataFrame({'choices': x, 'similarity': y}, index=range(len(x)))

        # df = pd.DataFrame(list(results[k].items()), columns=['Key', 'Value']).transpose()
        # Save to Excel
        # df.to_excel(os.path.join(results_path + 'Semantics.xlsx'), index=False)
        # df.to_excel(os.path.join(results_path, 'Semantics.xlsx'), sheet_name=k, index=False, header=False)

        with pd.ExcelWriter(os.path.join(results_path, 'Semantics.xlsx'), engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=k, index=False)




if __name__ == '__main__':
    main()
