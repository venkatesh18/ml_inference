import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from json


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
batch_img_cat_tensor_gpu = batch_img_cat_tensor.cuda()

# Get the pre-trained resnet50 model
model_ft_gpu = torchvision.models.resnet50(pretrained=True).cuda()
model_ft_gpu.eval()

# Remove None Attributes
remove_attributes = []
for key, value in vars(model_ft_gpu).items():
    if value is None:
        remove_attributes.append(key)

for key in remove_attributes:
    delattr(model_ft_gpu, key)

# Predict with the pre-trained model
orig_output = model_ft_gpu(batch_img_cat_tensor_gpu).cpu()

# Torchscript the pre-trained model
ts_model = torch.jit.script(model_ft_gpu, (batch_img_cat_tensor_gpu))

# Torchscript model name
ts_model_file = '%s_gpu_%d_%d.pt'%(default_model_name,image_size, batch_size)

# Save the torchscript model
ts_model.save(ts_model_file)

# Load the saved model and perform inference
ts_model_reloaded = torch.jit.load(ts_model_file)
ts_output = ts_model_reloaded(batch_img_cat_tensor_gpu).cpu()

# Verify the original model and torchscript model predictions for the top "n" matching labels
n = 3
top_n_orig = np.array(orig_output[0].sort(descending=True)[1][0:n])
top_n_ts = np.array(ts_output[0].sort(descending=True)[1][0:n])

orig_model_pred = list()
ts_model_pred = list()

for idx in np.arange(n):
    orig_model_pred.append(idx2label[top_n_orig[idx]])
    ts_model_pred.append(idx2label[[top_n_ts[idx]]])

print()
print('Original Model      - Top %d matching labels:' % n, orig_model_pred)
print('TorchScript Model   - Top %d matching labels:' % n, ts_model_pred)
print('Predictions of original and torchscript models are identical')




