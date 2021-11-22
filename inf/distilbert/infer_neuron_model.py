import os
import numpy as np
import time
import random
import torch
import subprocess
import torch.neuron
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from essential_generators import DocumentGenerator


# Benchmark test parameters - Number of models, threads, total number of requests
num_models = 1  # num_models <= number of cores (4 for inf1.xl and inf1.2xl, 16 for inf1.6xl)
num_threads = num_models * 1  # Setting num_threads to num_models works well.
num_requests = 10000

# Input parameters : token length, batch size
max_length = 64
batch_size = 1
model_name = 'distilbert-base-uncased'

total_sentences = num_requests * batch_size
print('Benchmark Test Parameters')
print('Input batch Size = %d' % batch_size)
print('Number of requests = %d' % num_requests)
print('Total number of sentences (num_requests x batch_size) = %d' % total_sentences)

# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
num_neuron_chips = int(subprocess.getoutput('ls /dev/neuron* | wc -l'))
num_cores = 4 * num_neuron_chips
nc_env = ','.join(['1'] * num_cores)
print('Neuron Core Group Sizes: %s'%(nc_env))
os.environ['NEURONCORE_GROUP_SIZES'] = nc_env
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

# Neuron file name
neuron_model_file = '%s_inf_%d_%d.pt'%(model_name, max_length, batch_size)

# Get tokenizer and create encoded inputs
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate sample sentences
gen = DocumentGenerator()
sequence_list = []
encoded_input_list = []
num_samples = 100

for _ in np.arange(num_samples):
    sequence = gen.sentence()
    encoded_inputs = tokenizer.encode_plus(sequence, max_length=max_length, padding='max_length', truncation=True,
                                           return_tensors='pt')
    sequence_list.append(sequence)
    encoded_input_list.append(encoded_inputs)

# Function to load the model
def load_model(file_name):
    model = torch.jit.load(file_name)

    return model

latency_list = []

# Function for model prediction and for measuring latency
def task(model, encoded_inputs):
    global latency_list
    begin = time.time()

    input_ids_tensor = encoded_inputs['input_ids']
    batch_input_ids_tensor = torch.cat([input_ids_tensor] * batch_size)
    attention_mask_tensor = encoded_inputs['attention_mask']
    batch_attention_mask_tensor = torch.cat([attention_mask_tensor] * batch_size)
    ts_input = batch_input_ids_tensor, batch_attention_mask_tensor

    _ = model(*ts_input)
    latency_time = time.time() - begin
    latency_list.append(latency_time)

    return

# Function for running benchmark
def benchmark(num_models, num_threads, num_requests, model_file):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file) for _ in range(num_models)]
    print('Starting benchmark')
    # output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                futures.append(pool.submit(task, models[i % len(models)], random.choice(encoded_input_list)))
                # output_list.append(output.result())
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    test_time = time.time() - begin

    # return test_time, np.array(output_list)
    return test_time


# test_time, latency_array = benchmark(num_models, num_threads, num_requests, neuron_model_file)
test_time = benchmark(num_models, num_threads, num_requests, neuron_model_file)
latency_metrics = np.percentile(np.array(latency_list) * 1000, [50, 90, 95])
print('Latency: [P50, P90, P95] milli seconds')
print(np.round(latency_metrics, 3), 'milli seconds')
print('Total time taken for %d sentences is %0.2f seconds' % (total_sentences, test_time))
print('Throughput (sentences /sec) = %0.2f' % (total_sentences / test_time))
