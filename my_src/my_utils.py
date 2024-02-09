import os
import shutil
import json
import math
import numpy as np
import pandas as pd
import torch
from open_clip import tokenizer
# import openai
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')  # Backend image engine that works in pycharm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def loadCaptions(file_path):
    # Open the file and load the JSON data
    try:
        with open(file_path, 'r') as json_file:
            captions = json.load(json_file)
            # print(captions)  # This will print the data loaded as a dictionary
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"There was an error decoding the JSON from {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return captions



def findCaptionForImage(imagePath, captions):

    my_images = {
        'names': [],
        'captions': {},
        'imageIds': []
    }

    imageFileNames = [filename for filename in os.listdir(imagePath) if
                     filename.endswith(".png") or filename.endswith(".jpg")]

    for imageName in imageFileNames:
        # Find the matching caption
        image_id = None
        for img in captions['images']:
            if img['file_name'] == imageName:
                image_id = img['id']
                my_images['names'].append(imageName)
                my_images['imageIds'].append(image_id)
                break
        else:
            print('Image not found')
            image_id = None

        if image_id is not None:
            # Now find the corresponding caption for that image
            for ann in captions['annotations']:
                if ann['image_id'] == image_id:
                    print('Matching Caption:', ann['caption'])
                    if image_id in my_images['captions'].keys():
                        my_images['captions'][image_id].append(ann['caption'])
                    else:
                        my_images['captions'].update({image_id: [ann['caption']]})

        else:
            # Handle case where the image was not found
            print('Image was not found')

    return my_images


def plotImages(my_images, imagPath):

    plt.figure(figsize=(16, 5))
    numImages = len(my_images['names'])
    for i in range(numImages):
        image = Image.open(os.path.join(imagPath, my_images['names'][i])).convert("RGB")
        plt.subplot(math.ceil(numImages / 3), 3, i + 1)
        plt.imshow(image)
        plt.title(f"{my_images['names'][i]}\n{my_images['captions'][i]}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def plotSimilarity(similarity, original_images, texts, numImages):
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(numImages), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, numImages - 0.5])
    plt.ylim([numImages + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    plt.show()


def negateCaptions(myText):

    prompts = ['negate: ', 'negate one element: ', 'provide an opposite example for the following sentence: ']

    print('negateCaptions for: ' + myText)
    myResponse = []
    for myPrompt in prompts:
        # Generate a response from the model
        response = openai.Completion.create(
            # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
            model="gpt-3.5-turbo-instruct",
            prompt=myPrompt + myText,
            # n=3,
            max_tokens=60,  # Adjust based on how long you expect the response to be
        )

        # Extract the text from the response
        myResponse.append(response.choices[0].text.strip())

    print(myResponse)
    return myResponse


def loadTextFile(filePath):
    # Open the file and read its contents
    with open(filePath, 'r') as file:
        contents = file.read().split('\n')

    return contents


def saveResultsToExcel(results, resultsPath, fileName):
    filePath = os.path.join(resultsPath, fileName + '.xlsx')
    for k in list(results.keys()):
        # Convert to a DataFrame
        x = np.array(results[k]['choices'])
        y = np.array(results[k]['similarity'])
        df = pd.DataFrame({'choices': x, 'similarity': y}, index=range(len(x)))

        with pd.ExcelWriter(filePath, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=k, index=False)

    print('Saved results to: ' + filePath)


# Function to encode the image
def encodeImageBase64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def getModelResponse(openai, request):
    # Generate a response from the model
    response = openai.Completion.create(
        # engine="text-davinci-003",  # Replace with the appropriate model/engine for GPT-4 or the version you are using
        model="gpt-3.5-turbo-instruct",
        # model="gpt-4-vision-preview",
        prompt=request,
        max_tokens=100,  # Adjust based on how long you expect the response to be
    )
    # Extract the text from the response
    return response.choices[0].text.strip()


def printModelIo(runParams, text, isInput):
    action = "request" if isInput else "response"
    print("**** Model " + action + ": ***")
    print(text + "\n")

    with open(os.path.join(runParams.tempPath, runParams.promptFileTemp), 'a') as file:
        file.write("**** Model " + action + ": ***\n")
        file.write(text + "\n")


def cosineSimilarity(model, preprocess, captions, imagePath, imageName):

    text_tokens = tokenizer.tokenize(["This is " + desc for desc in captions])
    image = Image.open(os.path.join(imagePath, imageName)).convert("RGB")
    image_input = torch.tensor(np.stack([preprocess(image)]))

    with torch.no_grad():
        # image_features = model.encodeImageBase64(image_input).float()
        # image_features = base64.b64encode(image_input)
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    ## Calculate cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    ## Display cosine similarity
    # plotSimilarity(similarity, [image], choices, 4)

    simDict = dict()
    simDict.update({'choices': captions})
    simDict.update({'similarity': similarity.squeeze().tolist()})

    return simDict


# Copy chatGPT log to results dir
def saveChatPrompt(runParams):
    promptFileTemp = os.path.join(runParams.tempPath, runParams.promptFileTemp)
    promptFile = os.path.join(runParams.resultsDir, 'chatGptPrompt.log')

    try:
        shutil.move(promptFileTemp, promptFile)
    except IOError as e:
        print(f"Unable to move file. {e}")


def saveImagesWithCaptions(my_images, results, runParams):

    for i in range(len(my_images['imageIds'])):
        imageId = my_images['imageIds'][i]
        imageName = my_images['names'][i]

        # Load existing image
        image = Image.open(os.path.join(runParams.imagePath, imageName)).convert("RGB")
        img_width, img_height = image.size

        # Text to be added
        choices = results.get(imageName).get('choices')
        scores = results.get(imageName).get('similarity')
        resLines = [" Score: " + "{:.3f}".format(round(scores[j], 3)) + ". " + choices[j] for j in range(len(choices))]

        # Sort the texts in descending order of scores, keeping track of indices (original true sentence was last)
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        indices = [index for index, value in sorted_indices]
        resLines = [resLines[ind] for ind in indices]

        # Create new image with space for text
        new_image = Image.new("RGB", (img_width + 300, img_height + 300), "white")
        new_image.paste(image, (0, 0))

        # Draw on the new image
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("arial.ttf", 15)

        # Text parameters
        x = 10  # Horizontal coordinate to start
        y = img_height + 10  # Vertical coordinate to start
        line_height = 20  # Distance between lines of text

        text = "Cosine similarity test:"
        draw.text((x, y), text, fill='black', font=font)
        y += line_height

        # Add choices texts
        for ind, text in enumerate(resLines):
            # Highlight original true sentence
            if indices[ind] == len(scores) - 1:
                color = 'green'
            else:
                color = 'black'
            draw.text((x, y), text, fill=color, font=font)
            y += line_height

        # Add original correct captions
        y += line_height
        text = "Original true captions:"
        draw.text((x, y), text, fill='black', font=font)
        y += line_height

        origCaptions = my_images.get('captions').get(imageId)
        for caption in origCaptions:
            # Highlight original true sentence
            draw.text((x, y), caption, fill='black', font=font)
            y += line_height


        # Save the new image
        outcome = 'right/' if (indices[0] == len(scores) - 1) else 'wrong/'  # get right/wrong result
        newImagePath = os.path.join(runParams.resultsDir + outcome, imageName)
        new_image.save(newImagePath)
        print('Saved results to: ' + newImagePath)


def calcMetrics(runParams, my_images, results):
    # Metrics
    right, wrong = 0, 0
    ranks = []
    for i in range(len(my_images['imageIds'])):
        imageId = my_images['imageIds'][i]
        imageName = my_images['names'][i]

        scores = results.get(imageName).get('similarity')
        # Sort the scores in descending order, keeping track of indices (original true sentence was last)
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        indices = [index for index, value in sorted_indices]

        rank = indices.index(len(scores) - 1)
        ranks.append(rank+1)
        if rank == 0:
            right = right + 1
        else:
            wrong = wrong + 1

    # Success rate
    successRate = 100 * right / (right + wrong)
    # Mean rank
    meanRank = np.array(ranks).mean()

    summaryText =f""" 
    Summary
    -------
    Right choice: {right}
    Wrong choice: {wrong}
    Success rate: {successRate:.2f}%
    Mean rank: {meanRank:.2f}
    """
    print(summaryText)

    with open(os.path.join(runParams.resultsDir, 'summary.txt'), 'a') as file:
        file.write(summaryText + "\n")

    metrics = dict()
    metrics.update({'right': right, 'wrong': wrong, 'ranks': ranks})

    return metrics
