import torch
import open_clip
from PIL import Image
import os
import numpy as np
import time


dataSetPath = '../../Dataset/'
runFolder = 'val2014/'
imagePath = dataSetPath + runFolder
encodedDir = dataSetPath + 'Encoded/' + runFolder

print('Encode images offline on: ' + runFolder)
start_time = time.time()

if not os.path.exists(encodedDir):
    os.makedirs(encodedDir)
    print(f'Created Encoded directory: ' + encodedDir)

## Load model
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model.eval()

imageFileNames = [filename for filename in os.listdir(imagePath) if
                     filename.endswith(".png") or filename.endswith(".jpg")]

for imageName in imageFileNames:

    image = Image.open(os.path.join(imagePath, imageName)).convert("RGB")
    image_input = torch.tensor(np.stack([preprocess(image)]))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

        # Save the tensor to a file
        tensorFileName = ('tensor_' + imageName).replace('.jpg', '.pt').replace('.png', '.pt')
        torch.save(image_features, os.path.join(encodedDir, tensorFileName))

end_time = time.time()
duration = end_time - start_time
print(f"Execution time : {duration / 60 :.2f} minutes")
