# This code is an adaptation, please follow ImageBind implementation
# (https://github.com/facebookresearch/ImageBind/tree/main)
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import pandas as pd
import tqdm
import numpy

audio_dir = "./data/DSAV360/audio_segments"
out_dir = "./data/DSAV360/audio_embeddings"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
for video_name in tqdm.tqdm(os.listdir(audio_dir)):
    video_audio_path = os.path.join(audio_dir, video_name)
    out_path = os.path.join(out_dir, video_name + ".csv")
    print(out_path)
    if os.path.exists(out_path):
        continue
    
    segment_paths = []
    frame_numbers = []
    embeddings = []
    with torch.no_grad():
        for video_segment in os.listdir(video_audio_path):
            segment_path = os.path.join(video_audio_path, video_segment)
            frame_numbers.append(video_segment.split('.')[0][-4:])
            
            input = {ModalityType.AUDIO: data.load_and_transform_audio_data([segment_path], device)}
            # Get the embeddings
            embedding = model(input)[ModalityType.AUDIO].numpy(force=True)
            embedding = numpy.squeeze(embedding)
            embeddings.append(embedding)
            #print(embedding[ModalityType.AUDIO].numpy(force=True))
    
    embeddings_to_write = pd.DataFrame(embeddings, index=frame_numbers)
    embeddings_to_write.to_csv(out_path, header=False)
