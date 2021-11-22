import numpy as np
import os
import time
import torch
import subprocess
import torch.neuron
from PIL import Image
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from tqdm import tqdm


# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
# num_cores = 4 # This value should be 4 on inf1.xlarge and inf1.2xlarge
num_neuron_chips = int(subprocess.getoutput('ls /dev/neuron* | wc -l'))
num_cores = 4 * num_neuron_chips
nc_env = ','.join(['1'] * num_cores)
print('Neuron Core Group Sizes: %s'%(nc_env))
os.environ['NEURONCORE_GROUP_SIZES'] = nc_env
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Benchmark test parameters - Number of models, threads, total number of requests
num_models = 1  # num_models <= number of cores (4 for inf1.xl and inf1.2xl, 16 for inf1.6xl)
num_threads = num_cores * 1  # Setting num_threads to num_models works well.
num_requests = 5000

# Set image size and input batch size
image_size = 224
batch_size = 4
total_images = num_requests * batch_size

print('Benchmark Test Parameters')
print('Image Size = %d x %d' % (image_size, image_size))
print('Input Batch Size = %d' % batch_size)
print('Number of requests = %d' % num_requests)
print('Total number of images (num_requests x batch_size) = %d' % total_images)

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
img_file_list = [x for x in img_file_list if '.jpg' in x]
num_images = len(img_file_list)

# Preprocess the images
for cur_image_file in img_file_list:
    cur_image = Image.open('%s/%s' % (data_dir, cur_image_file)).convert('RGB')
    cur_image_preprocessed = preprocess(cur_image)
    cur_image_preprocessed_unsqueeze = torch.unsqueeze(cur_image_preprocessed, 0)
    img_preprocessed_list.append(cur_image_preprocessed_unsqueeze)


# Neuron file name
model_name = 'resnet50'
neuron_model_file = '%s_inf_%d_%d.pt'%(model_name,image_size, batch_size)

# Function to load the model
def load_model(file_name):
    # Load modelbase
    model = torch.jit.load(file_name)

    return model

latency_list = []

# Function for model prediction and for measuring latency
def task(model, cur_img_preprocess):
    global latency_list
    begin = time.time()

    batch_input_tensor = torch.cat([cur_img_preprocess] * batch_size)

    prediction = model(batch_input_tensor)
    latency_time = time.time() - begin

    latency_list.append(latency_time)
    return

# Function for running benchmark
def benchmark(num_models, num_threads, num_requests, model_file):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file) for _ in range(num_models)]
    print('Starting benchmark')
    #output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                futures.append(pool.submit(task, models[i % len(models)], img_preprocessed_list[i % num_images]))
                #output_list.append(output.result())
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    test_time = time.time() - begin

    # return test_time, np.array(output_list)
    return test_time


# test_time, latency_array = benchmark(num_models, num_threads, num_requests, neuron_model_file)
test_time = benchmark(num_models, num_threads, num_requests, neuron_model_file)
latency_metrics = np.percentile(np.array(latency_list), [50, 90, 95])
print('Latency: [P50, P90, P95] milli seconds')
print(np.round(latency_metrics, 3), 'milli seconds')
print('Total time taken for %d images is %0.4f seconds' % (total_images, test_time))
print('Throughput (images /sec) = %0.4f' % (total_images / test_time))
