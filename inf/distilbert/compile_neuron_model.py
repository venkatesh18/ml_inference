import os
import time
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification


os.environ['TOKENIZERS_PARALLELISM'] = 'False'

# Input parameters
max_length = 64   # Maximum token length
batch_size = 4     # Input batch size
model_name = 'distilbert-base-uncased' # model name

print('Model name: %s' % model_name)
print('Input batch size = %d' % batch_size)
print('Maximum token length = %d' % max_length)

# Sample sentence
sequence = 'I am going to a movie'

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

# Create input tuple
sample_input = batch_input_ids_tensor, batch_attention_mask_tensor

# Get the pre-trained model 
orig_model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=False)

# Predict with the pre-trained model
orig_output = orig_model(*sample_input)

# Compile the model with torch neuron
ts = time.time()
print('Starting model compilation')
print()
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print()

# Compile the model
neuron_model = torch.neuron.trace(orig_model, sample_input)

# Save the compiled model for later use
neuron_model_file = '%s_inf_%d_%d.pt'%(model_name, max_length, batch_size)
neuron_model.save(neuron_model_file)
telapsed = time.time() - ts

print()
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print()
print('Time for compiling the DISTILBERT-BASE-UNCASED model = %0.2f seconds' % telapsed)

# Load the saved model and perform inference
neuron_model_reloaded = torch.jit.load(neuron_model_file)
neuron_output = neuron_model_reloaded(*sample_input)

print('Original Model Output:', orig_output)
print('Neuron Model Output:  ', neuron_output)

