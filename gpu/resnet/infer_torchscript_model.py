import os
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Benchmark test parameters
num_requests = 5000
num_models = 1
num_threads = num_models * 1
mixed_precision = True

# Image size and input batch size
image_size = 224
batch_size = 4
total_images = num_requests * batch_size

print('Benchmark Test Parameters')
print('Image Size = %d x %d' % (image_size, image_size))
print('Input Batch Size = %d' % batch_size)
print('Number of requests = %d' % num_requests)
print('Total number of images (num_requests x batch_size) = %d' % total_images)
print('Mixed Precision = ', mixed_precision)

# Create a preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# Load images from the data folder
data_dir = './data'
img_preprocessed_list = list()
img_file_list = os.listdir(data_dir)
img_file_list = [x for x in jpg_file_list if '.jpg' in x]
num_images = len(img_file_list)

# Preprocess the images
for cur_image_file in img_file_list:
    cur_image = Image.open('%s/%s' % (data_dir, cur_image_file)).convert('RGB')
    cur_image_preprocessed = preprocess(cur_image)
    cur_image_preprocessed_unsqueeze = torch.unsqueeze(cur_image_preprocessed, 0)
    img_preprocessed_list.append(cur_image_preprocessed_unsqueeze)

# Torchscript model name
ts_model_file = 'resnet50_gpu_%d_%d.pt'%(image_size, batch_size)

# Function to load the model
def load_model(file_name, torchscript):
    with torch.cuda.amp.autocast(enabled=half_precision):
        if torchscript:
            model = torch.jit.load(file_name)
            model.eval()
            model = model.cuda()
        else:
            model = torchvision.models.resnet50(pretrained=True)
            model.eval()
            model = model.cuda()

    return model

latency_list = []

# Function for model prediction and for measuring latency
def task(model, cur_img_preprocess):
    global latency_list
    begin = time.time()
    with torch.cuda.amp.autocast(enabled=half_precision):
        batch_input_tensor = torch.cat([cur_img_preprocess] * batch_size)
        batch_input_tensor_gpu = batch_input_tensor.cuda()
        prediction = model(batch_input_tensor_gpu)
        latency_time = time.time() - begin

        latency_list.append(latency_time)
    return

# Function for running benchmark
def benchmark(num_models, num_threads, num_requests, model_file, torchscript=True):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file, torchscript) for _ in range(num_models)]
    print('Starting benchmark')
    output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                futures.append(pool.submit(task, models[i % len(models)], random.choice(img_preprocessed_list)))
                #output_list.append(output.result())
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    test_time = time.time() - begin

    # return test_time, np.array(output_list)
    return test_time


test_time = benchmark(num_models, num_threads, num_requests, ts_model_file, torchscript=True)
latency_metrics = np.percentile(np.array(latency_list), [50, 90, 95])
print('Latency: [P50, P90, P95] milli seconds')
print(np.round(latency_metrics, 3), 'milli seconds')
print('Total time taken for %d images is %0.4f seconds' % (total_images, test_time))
print('Throughput (images /sec) = %0.4f' % (total_images / test_time))
