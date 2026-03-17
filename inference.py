import os
import numpy as np
import torch
import config
from dataloader import AV_Dataloader
from torch.utils.data import DataLoader
import cv2
import tqdm
from utils import read_txt_file, save_video
from model import AViSal360



def eval(test_data, model, device, result_imp_path):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, aem, emb, names in tqdm.tqdm(test_data):

            pred = model(x.to(device), aem.to(device), emb.to(device))

            batch_size, _,_ = pred[:, 0, :, :].shape
            
            for bs in range(batch_size):
     
                folder = os.path.join(result_imp_path, names[-1][bs].split('_')[0])
                if not os.path.exists(folder):
                    os.makedirs(folder)

                sal = pred[bs, 0, :, :].cpu()
                sal = np.array((sal - torch.min(sal)) / (torch.max(sal) - torch.min(sal)))
                cv2.imwrite(os.path.join(folder, names[-1][bs] + '.png'), (sal * 255).astype(np.uint8))


if __name__ == "__main__":

    video_test_names = read_txt_file(config.videos_test_file)
    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device") 

    model_parameters = torch.load(config.inference_model, map_location=device)
    avisal360 = AViSal360(**model_parameters["model_args"])
    avisal360.load_state_dict(model_parameters["model_state_dict"])
    

    # Load the data. Use the appropiate data loader depending on the expected input data
    test_video360_dataset = AV_Dataloader(config.frames_dir, None, 
                                          path_to_embeddings= config.embeddings_dir, 
                                          path_to_AEM=config.audio_AEM_dir, 
                                          video_names=video_test_names, 
                                          frames_per_data=config.sequence_length, 
                                          split='test', 
                                          load_names=True, 
                                          HxW_frames = config.resolution,
                                          HxW_AEM=(120,160),
                                          inter_seq=True,
                                          random_embeddings=False,
                                          window=1)

    test_data = DataLoader(test_video360_dataset, batch_size=config.batch_size, num_workers=10, shuffle=False)

    eval(test_data, avisal360, device, config.results_dir)

    # Save video with the results
    
    if config.save_videos:
        
        for video_name in video_test_names:
            save_video(os.path.join(config.frames_dir, video_name), 
                    os.path.join(config.results_dir, video_name),
                    None,
                    'AViSal_pred_' + video_name +'.avi')