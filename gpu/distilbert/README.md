# Instructions for torchscripting and performing inference

```
This file provides the instructions for torchscripting the pre-trained DISTILBERT BASE UNCASED MNLI model and for 
performing inference with the torchscripted model. The python scripts in this example should be executed on an 
EC2 g4dn.2xlarge instance, with Ubuntu 18:04 DLAMI.
```

## Setting the pytorch environment

```
On a g4dn.2xlarge instance, type 'conda env list' on the terminal to list all the available environments.
One of the available environments would be 'pytorch_latest_p37'.

Execute the command below in the terminal to activate a pytorch environment.
'conda activate pytorch_latest_p37'

If this environment is unavailable, choose the available pytorch environment.
```

## Torchscript the model

```
To torchscript the DISTILBERT model, run 'create_torchscript_model.py' with the command
'python create_torchscript_model.py'

The torchscript model is stored in the same folder that contains this file.
```

## Run inference with the torchscript model

```
To perform inference, run 'infer_torchscript_model.py' with the command
'python infer_torchscript_model.py'
