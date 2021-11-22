import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Input parameters
max_length = 64   # Maximum token length
batch_size = 1    # Input batch size
model_name = 'distilbert-base-uncased' # model name

print('Model name: %s' % model_name)
print('Input batch size = %d' % batch_size)
print('Maximum token length = %d' % max_length)

sequence = 'I am going to a movie'
ts_model_file = '%s_gpu_%d_%d.pt'%(model_name, max_length, batch_size)

# Get tokenizer and create encoded inputs
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded_inputs = tokenizer.encode_plus(sequence, max_length=max_length, padding='max_length', truncation=True,
                                       return_tensors='pt')
# Get input ids and attention mask
input_ids_tensor = encoded_inputs['input_ids']
attention_mask_tensor = encoded_inputs['attention_mask']

# Multiply by batch size
batch_input_ids_tensor = torch.cat([input_ids_tensor] * batch_size)
batch_attention_mask_tensor = torch.cat([attention_mask_tensor] * batch_size)

# Get the model and predict
orig_model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)

# Push model into cuda
orig_model_cuda = orig_model.cuda()

# create input tuple
ts_input = batch_input_ids_tensor.cuda(), batch_attention_mask_tensor.cuda()

# predict
orig_output = orig_model_cuda(*ts_input)

# Torchscript the model
ts_model = torch.jit.trace(orig_model_cuda, ts_input)

# Save the compiled model for later use
ts_model.save(ts_model_file)

# Load the saved torchscript model and perform inference
ts_model_reloaded = torch.jit.load(ts_model_file)
ts_output = ts_model_reloaded(*ts_input)

print('Original Model Output:   ', orig_output)
print('Torchscript Model Output:', ts_output)
