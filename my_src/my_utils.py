import os
import json
import math
import openai
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')  # Backend image engine that works in pycharm
import matplotlib.pyplot as plt

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
        'captions': [],
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
                    my_images['captions'].append(ann['caption'])
                    break
            else:
                print('Caption not found')
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
    openai.api_key = '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'
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
