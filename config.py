#####################################################################################
# Data Loader parameters
#####################################################################################
# Sequence length
sequence_length = 20
# Image resolution (height, width)
resolution = (240, 320)
# Path to the folder containing the RGB frames
frames_dir = 'data/frames'
# Path to the folder containing the AEMs
optical_flow_dir = 'data/optical_flow'
# Path to the folder containing the AEMs
audio_AEM_dir = 'data/AEM'
# Path to the folder containing the ImageBind embeddings
embeddings_dir = 'data/embeddings'
# Path to the folder containing the ground truth saliency maps
gt_dir = 'data/saliency_maps'
# Folder containing the kfold splits
k_folds_dir = 'k_folds'

#####################################################################################
# Training parameters
#####################################################################################
# Batch size
batch_size = 1
# Nº of epochs
epochs = 120
# Learning rate
lr = 0.8
# Hidden dimension of the model
hidden_dim = 36
# Name of the model
model_name = 'AViSal360'
# Path to the folder where the checkpoints will be saved
ckp_dir = 'checkpoints'
# Path to the folder where the model will be saved
models_dir ='models'
# Path to the folder where the training logs will be saved
runs_data_dir = 'runs'
