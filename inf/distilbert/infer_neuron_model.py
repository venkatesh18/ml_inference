import os
import torch
import subprocess
import torch.neuron
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import time
import random
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from essential_generators import DocumentGenerator

# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
num_neuron_chips = int(subprocess.getoutput('ls /dev/neuron* | wc -l'))
num_cores = 4 * num_neuron_chips
nc_env = ','.join(['1'] * num_cores)
print('Neuron Core Group Sizes: %s'%(nc_env))
os.environ['NEURONCORE_GROUP_SIZES'] = nc_env
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

# Benchmark test parameters - Number of models, threads, total number of requests
num_models = 1  # num_models <= number of cores (4 for inf1.xl and inf1.2xl, 16 for inf1.6xl)
num_threads = num_models * 1  # Setting num_threads to num_models works well.
num_requests = 10000


# Input parameters : token length, batch size
max_length = 64
batch_size = 4
model_name = 'distilbert-base-uncased'


# Neuron file name
neuron_model_file = '%s_inf_%d_%d.pt'%(model_name, max_length, batch_size)



# Create a pipeline with the given model
model_dict = dict()
model_dict['return_dict'] = False
# Get tokenizer and create encoded inputs
tokenizer = AutoTokenizer.from_pretrained(model_name)

gen = DocumentGenerator()
sequence_list = []
encoded_input_list = []
num_request_samples = 10

for _ in np.arange(num_request_samples):
    sequence = gen.sentence()
    encoded_inputs = tokenizer.encode_plus(sequence, max_length=max_length, padding='max_length', truncation=True,
                                           return_tensors='pt')
    sequence_list.append(sequence)
    encoded_input_list.append(encoded_inputs)


def load_model(file_name):
    # Load modelbase
    model = torch.jit.load(file_name)

    return model

latency_list = []

def task(model, encoded_inputs):
# def task(model, tokeniz, sentence):
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


def benchmark(num_models, num_threads, num_requests, model_file):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file) for _ in range(num_models)]
    print('Starting benchmark')
    output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    # https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                futures.append(pool.submit(task, models[i % len(models)], random.choice(encoded_input_list)))
                # output_list.append(output.result())
            print('Loaded Requests')
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    test_time = time.time() - begin

    # return test_time, np.array(output_list)
    return test_time


# test_time, latency_array = benchmark(num_models, num_threads, num_requests, neuron_model_file)
test_time = benchmark(num_models, num_threads, num_requests, neuron_model_file)
print('Latency: %d samples: (P50, P90, P95)'%(len(latency_list)))
print(np.percentile(np.array(latency_list), [50, 90, 95]))
print('Total time taken for %d * (%d x sentences) is %0.4f seconds' % (num_requests, batch_size, test_time))
print('Throughput (sentences * batch_size /sec) = %0.4f' % (num_requests * batch_size/ test_time))
# print(bench_output[100])
