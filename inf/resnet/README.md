# Instructions for compiling and performing inference

```
This file provides the instructions for compiling the pre-trained RESNET50 model, with torch neuron, 
and for performing inference with the compiled model. The python scripts in this example should be 
executed on an EC2 inf1.2xlarge instance, with Ubuntu 18:04 DLAMI.
```

## Setting the neuron pytorch environment

```
On a inf1.2xlarge instance, type 'conda env list' on the terminal to list the available environments.
One of the available environments would be 'aws_neuron_pytorch_p36'.

Execute the command below in the terminal to activate this neuron pytorch environment.
'conda activate aws_neuron_pytorch_p36'

If this environment is not present, activate the available neuron pytorch environment.
```

## Compile the model

```
To compile the RESNET50 model, run 'compile_neuron_model.py' with the command
'python compile_neuron_model.py'

The model compilation may take 2 to 3 minutes.

The compiled model is stored in the same folder that contains this file.
```

## Run inference with the compiled model

```
To perform inference, run 'infer_neuron_model.py' with the command
'python infer_neuron_model.py'
