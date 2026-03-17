import os 
import pandas as pd
import numpy as np
import config

files_path = config.csv_results_files_path
metrics = config.metrics_to_plot
outlier_remove = config.outlier_remove
models_to_use = config.models_to_use

# Get the list of cvs files in the folder
files = [f for f in os.listdir(files_path) if f.endswith('.csv')]

# Read all the cvs files in the folder and concatenate them adding a column with the name of the file
df = pd.concat([pd.read_csv(os.path.join(files_path, f)).assign(model=f[:-4]) for f in files])

df = df[df['model'].isin(models_to_use)]

# Get the list of models
models = df['model'].unique()
videos = df['Video'].unique()

print('Video len:', len(videos))
# Check if the frames are the same for all the models
frames = df['Frame'].unique()


for model in models:
    frames_model = df.loc[df['model'] == model, 'Frame'].unique()
    if len(frames_model) != len(frames):
        print("ERROR: The number of frames is not the same for all the models")
        # Print the frames that are not in frames_model
        print('Total frames: ', len(frames))
        print('Removing: ', len(list(set(frames).symmetric_difference(set(frames_model)))), 'from model', model)
        # Print the number of "frames_model" element not present in "frames"
        print('Nº missing frames: ', len(list(set(frames_model).difference(set(frames)))))
        print(list(set(frames_model).difference(set(frames)))[0:20])
        # Print the number of "frames" element not present in "frames_model"
        print('Nº additional frames: ', len(list(set(frames).difference(set(frames_model)))))
        print(list(set(frames).difference(set(frames_model)))[0:20])

        # Remove the frames that are not in all the models
        df = df[~df['Frame'].isin(list(set(frames).symmetric_difference(set(frames_model))))]
        frames = df['Frame'].unique()


scores_df = pd.DataFrame(columns=['model', 'Video',  'AUC_Judd_mean', 'AUC_Judd_std', 'NSS_mean', 'NSS_std', 'CC_mean', 'CC_std', 'SIM_mean', 'SIM_std', 'KLD_mean', 'KLD_std', 'EMD_mean', 'EMD_std', 'MAE_mean', 'MAE_std', 'MSE_mean', 'MSE_std'])

for model in models:
    for video in videos:
        df_video = df.loc[(df['Video'] == video) & (df['model'] == model), metrics]
        score = pd.DataFrame(columns=['model', 'Video', 'AUC_Judd_mean', 'AUC_Judd_std','NSS_mean', 'NSS_std', 'CC_mean', 'CC_std', 'SIM_mean', 'SIM_std', 'KLD_mean', 'KLD_std', 'EMD_mean', 'EMD_std', 'MAE_mean', 'MAE_std', 'MSE_mean', 'MSE_std'])
        score['model'] = [model]
        score['Video'] = [video]
        for metric in metrics:
            if outlier_remove:
                # Remove the outliers with 3 IQR rule
                Q1 = df_video[metric].quantile(0.25)
                Q3 = df_video[metric].quantile(0.75)
                IQR = Q3 - Q1
                df_video = df_video[~((df_video[metric] < (Q1 - 1.5 * IQR)) | (df_video[metric] > (Q3 + 1.5 * IQR)))]
                # print('Removing outliers: ', len(df_video))

            if metric == 'MSE':
                # Apply the square root to the MSE
                df_video[metric] = np.sqrt(df_video[metric])
            
            score[metric + '_mean'] = [df_video[metric].mean()]
            score[metric + '_std'] = [df_video[metric].std()]
        scores_df = pd.concat([scores_df, score], ignore_index=True)
      
# Plot the metrics for each model with violin plots
for metric in metrics:
    for idx, model in enumerate(models):
        scores_model = scores_df.loc[scores_df['model'] == model, metric + '_mean']
        scores_model_std = scores_df.loc[scores_df['model'] == model, metric + '_std']
        print('Model: ', model, ',   Metric: ', metric, ',   Mean: ', scores_model.mean(), ',   Std: ', scores_model.std())

