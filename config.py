#####################################################################################
# Data Loader parameters
#####################################################################################
# Sequence length
sequence_length = 20
# Image resolution (height, width)
resolution = (240, 320)
# Path to the folder containing the RGB frames
frames_dir = '.data/frames_8fps_HR'
# Path to the folder containing the AEMs
audio_AEM_dir = '.data/AEM'
# Path to the folder containing the ImageBind embeddings
embeddings_dir = '.data/audio_embeddings'
# Path to the folder containing the ground truth saliency maps
gt_dir = '.data/saliency_maps_8fps'
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

#####################################################################################
# Inference parameters
#####################################################################################
# Path to the folder containing the model to be used for inference
inference_model = './models/AViSal360_train_fold0.pth'
# Path to the folder where the inference results will be saved
results_dir = './AViSal360_DSAV360_predicted_salmaps'
# Path to the txt file containing the videos to be used for inference
videos_test_file = './k_folds/test_0.txt' # Use the same number (the videos in test_0.txt are the ones not present in train_0.txt)
save_videos = False

# ==================================================================================================
# Saliency map metric evaluation parameters (compute_metrics.py):
# ==================================================================================================
predicted_salmaps_path = "./AViSal360_DSAV360_predicted_salmaps" # Path to the folder containing the predicted saliency maps
gt_salmap_path = "./data/saliency_maps_8fps" # Path to the folder containing the ground truth saliency maps
gt_fixations_file_path = "./data/fixations_8fps" # Path to the folder containing the ground truth fixations
output_cvs_file_path = "./AViSal_csv_results/AViSal360_DSAV360.csv" # Path to the output CSV file
salmaps_resolution = (320, 240) # (W,H)
metrics_to_compute = ['CC', 'SIM', 'NSS','MSE']
sampling_type = [ # Different sampling method to apply to saliency maps
"Sphere_9999999", # Too many points
"Sphere_1256637", # 100,000 points per steradian
"Sphere_10000",   # 10,000
"Sin",			  # Sin(height)
"Equi"			  # None
]
sampling_type = sampling_type[-2] # Sin weighting by default

# ==================================================================================================
# Plot metric evaluation parameters (plot_metrics_comparative.py):
# ==================================================================================================
csv_results_files_path = "./AViSal_csv_results"
metrics_to_plot = metrics_to_compute
outlier_remove = False
models_to_use = ['AViSal360_DSAV360']
