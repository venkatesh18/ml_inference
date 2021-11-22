import numpy as np
import json
import torch
import torch.neuron
from PIL import Image
import torchvision
from torchvision import transforms
import time


# Set image size and input batch size
image_size = 224
batch_size = 4
print('Image Size: %d x %d, Input Batch Size: %d'%(image_size, image_size, batch_size))

# Read the labels and create a list to hold them for classification
with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Open a sample image
img_cat = Image.open("data/cat_on_car.jpg").convert('RGB')

# Create a preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# Preprocess the sample image
img_cat_preprocessed = preprocess(img_cat)
img_cat_preprocessed_unsqueeze = torch.unsqueeze(img_cat_preprocessed, 0)
batch_img_cat_tensor = torch.cat([img_cat_preprocessed_unsqueeze] * batch_size)

# Get the pre-trained resnet50 model
model_ft = torchvision.models.resnet50(pretrained=True)
model_ft.eval()

# Remove None Attributes
remove_attributes = []
for key, value in vars(model_ft).items():
    if value is None:
        remove_attributes.append(key)

for key in remove_attributes:
    delattr(model_ft, key)

# Predict with the pre-trained model
orig_output = model_ft(batch_img_cat_tensor)

# Compile the model
model_name = 'resnet50'
neuron_model_file = '%s_inf_%d_%d.pt'%(model_name,image_size, batch_size)

ts = time.time()
print('Starting model compilation')
print()
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print()
neuron_model = torch.neuron.trace(model_ft, batch_img_cat_tensor)

# Save the compiled model for later use
neuron_model.save(neuron_model_file)
telapsed = time.time() - ts
print()
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print()
print('Time for compiling the RESNET50 model = %0.2f seconds' % telapsed)

# Load the saved model and perform inference
neuron_model_reloaded = torch.jit.load(neuron_model_file)
neuron_output = neuron_model_reloaded(batch_img_cat_tensor)

# Verify the original model and torchscript model predictions for the top "n" matching labels
n = 3
top_n_orig = np.array(orig_output[0].sort(descending=True)[1][0:n])
top_n_neuron = np.array(neuron_output[0].sort(descending=True)[1][0:n])

orig_model_pred = list()
neuron_model_pred = list()

for idx in np.arange(n):
    orig_model_pred.append(idx2label[top_n_orig[idx]])
    neuron_model_pred.append(idx2label[[top_n_neuron[idx]]])

print()
print('Original Model      - Top %d matching labels:' % n, orig_model_pred)
print('Neuron Model        - Top %d matching labels:' % n, neuron_model_pred)
print('Predictions of original and neuron models are identical')

