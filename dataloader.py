import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import librosa as lr
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as df
 
class AV_Dataloader(Dataset):
    def __init__(self, path_to_frames, path_to_saliency_maps, 
                 video_names=None, 
                 path_to_AEM=None,
                 path_to_OF=None, 
                 path_to_audio = None, 
                 path_to_logMel = None,
                 path_to_embeddings = None, 
                 frames_per_data=20, 
                 split_percentage=0.2, 
                 split='train', 
                 HxW_frames = [240, 320], 
                 HxW_AEM = [75, 130],
                 HxW_logMel = [240, 320],
                 skip=20, 
                 load_names=False, 
                 transform=False, 
                 inference=False, 
                 load_frames = True, 
                 random_AEM=False, 
                 random_wave=False, 
                 random_logMel=False,
                 random_embeddings=False,
                 inter_seq=False,
                 window=1,
                 threading=True):
        
        self.sequences = []
        self.correspondent_sal_maps = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.path_sal_maps = path_to_saliency_maps
        self.path_flow = path_to_OF
        self.resolution_frames = HxW_frames
        self.resolution_AEM = HxW_AEM
        self.resolution_logMel = HxW_logMel
        self.load_names = load_names
        self.transform = transform
        self.path_logMel = path_to_logMel
        self.path_audio = path_to_audio
        self.load_frames = load_frames
        self.loadlogMel =  self.path_logMel is not None
        self.logMel = []
        self.loadaudio = self.path_audio is not None
        self.audio = []
        self.path_AEM = path_to_AEM
        self.loadAEM = self.path_AEM is not None
        self.aems = []
        self.path_embeddings = path_to_embeddings
        self.load_embeddings = self.path_embeddings is not None
        self.load_OF = self.path_flow is not None
        self.embeddings = []
        self.threading = threading

        if random_AEM or random_wave or random_logMel:
            print('--------------------------------------------')
            print('--------------------------------------------')
            print('--------------------------------------------')
            print("WARNING: Random audio is activated.")
            print('--------------------------------------------')
            print('--------------------------------------------')
            print('--------------------------------------------')

        # Different videos for each split
        sp = int(math.ceil(split_percentage * len(video_names)))
        if split == "validation":
            video_names = video_names[:sp]
        elif split == "train":
            video_names = video_names[sp:]
        
        print('Loading data...')
        for name in tqdm(video_names):
            
            # Frames:
            video_frames_names = os.listdir(os.path.join(self.path_frames, name))
            video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))
            
            # AEMs:
            if self.loadAEM and random_AEM:
                # If random AEMs choose a random video and take the AEMs of that video
                random_video = np.random.choice(video_names)
                AEM_maps = os.listdir(os.path.join(self.path_AEM, random_video))  
                
            elif self.loadAEM:
                # If not random AEMs take the AEMs of the current video
                AEM_maps = os.listdir(os.path.join(self.path_AEM, name))
                AEM_maps = sorted(AEM_maps, key=lambda x: int((x.split(".")[0]).split("_")[1]))
            
            # Waveform :
            if self.loadaudio and random_wave:
                # If random audio choose a random video and take the audio of that video
                random_video = np.random.choice(video_names)
                random_audio_files = os.listdir(os.path.join(self.path_audio, random_video))

            # LogMel:   
            if self.loadlogMel and random_logMel:
                # If random logMel choose a random video and take the logMel of that video
                random_video = np.random.choice(video_names)
                logMel_maps = os.listdir(os.path.join(self.path_logMel, random_video))

            # Embeddings:
            if self.load_embeddings:
                # Load the embeddings
                df_embeddings = df.read_csv(os.path.join(self.path_embeddings, name + '.csv'), header=None)
                
                
            # -------------------- Create frames sequences --------------------
            
            # Frame number and saliency name must be the same (ex. frame name: 0001_0023.png, saliency map: 0023.png)

            # Skip the first frames to avoid biases due to the eye-tracking capture procedure 
            # (Observers are usually asked to look at a certain point at the beginning of each video )
            sts = skip

            # Split the videos in sequences of equal lenght
            initial_frame = self.frames_per_data + skip
            
            if inference:
                frames_per_data = self.frames_per_data - 4
            if inter_seq:
                frames_per_data = window

            # Split the videos in sequences of equal lenght
            for end in range(initial_frame, len(video_frames_names), frames_per_data):

                # Check if exist the ground truth saliency map for all the frames in the sequence
                valid_sequence = True

                for frame in video_frames_names[sts:end]:
                    if self.path_sal_maps is not None:
                        if not os.path.exists(os.path.join(self.path_sal_maps, frame.split("_")[0], frame)):
                            print('Saliency map not found for frame: ' + frame)
                            valid_sequence = False
                            break
                    if self.load_OF and not os.path.exists(os.path.join(self.path_flow, frame.split("_")[0], frame)):
                            print(os.path.join(self.path_flow, frame.split("_")[0], frame))
                            valid_sequence = False
                            print("Optical flow not found for frame: " + frame)
                            break
                
                last_frame = video_frames_names[end-1].split(".")[0]
                if self.load_embeddings and valid_sequence:
                    if df_embeddings.loc[df_embeddings.iloc[:, 0] == int(last_frame.split("_")[1])].empty:
                        valid_sequence = False
                        print('Embedding not found for frame: ' + last_frame)

                if valid_sequence:
                    # If the sequence is valid, append the frames to the list
                    self.sequences.append(video_frames_names[sts:end])
                    
                    # Waveform:
                    if self.loadaudio and random_wave:
                        # If random audio choose a random audio file
                        audio_sequence = random_audio_files[np.random.randint(0, len(random_audio_files))]
                        assert os.path.exists(os.path.join(self.path_audio, audio_sequence.split('_')[0], audio_sequence)), 'Audio file has not been found in path: ' + os.path.exists(os.path.join(self.path_audio, audio_sequence.split('_')[0], audio_sequence))
                        self.audio.append(audio_sequence)
                    elif self.loadaudio:
                        # Take the audio of the last frame of the sequence
                        audio_sequence = last_frame + ".wav"
                        assert os.path.exists(os.path.join(self.path_audio, name, audio_sequence)), 'Audio file has not been found in path: ' + os.path.exists(os.path.join(self.path_audio, name, audio_sequence))
                        self.audio.append(audio_sequence)
                    
                    # AEM:
                    if self.loadAEM and random_AEM:
                        # If random AEM choose a random AEM file
                        aem_sequence = [AEM_maps[np.random.randint(0, len(AEM_maps))] for _ in video_frames_names[sts:end]]
                        self.aems.append(aem_sequence)
                    elif self.loadAEM:
                        # Take the AEM of the last frame of the sequence
                        aem_sequence = [AEM_maps[int(frame.split("_")[1].split(".")[0])//8] for frame in video_frames_names[sts:end]] 
                        self.aems.append(aem_sequence)
                    
                    # LogMel:
                    if self.loadlogMel and random_logMel:
                        # If random logMel choose a random logMel file
                        logMel_sequence = logMel_maps[np.random.randint(0, len(logMel_maps))]
                        assert os.path.exists(os.path.join(self.path_logMel, logMel_sequence.split('_')[0], logMel_sequence)), 'LogMel file has not been found in path: ' + os.path.join(self.path_logMel, logMel_sequence.split('_')[0], logMel_sequence)
                        self.logMel.append(logMel_sequence)
                    elif self.loadlogMel:
                        # Take the logMel of the last frame of the sequence
                        logMel_sequence = last_frame + ".png"
                        assert os.path.exists(os.path.join(self.path_logMel, name, logMel_sequence)), 'LogMel file has not been found in path: ' + os.path.join(self.path_logMel, name, logMel_sequence)
                        self.logMel.append(logMel_sequence)
                        
                    # Embeddings:
                    if self.load_embeddings and random_embeddings:
                        # Take the embeddings of the last frame of the sequence
                        last_frame_number = int(last_frame.split("_")[1])
                        # If random embeddings create random embbedings of with random float numbers
                        assert not df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number].empty, 'Embedding not found for frame: ' + str(last_frame_number)+ '. Check the csv file.' + str(df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number, df_embeddings.columns[1:]].to_numpy())
                        shape_emb = (df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number, df_embeddings.columns[1:]].to_numpy()).shape
                        emb = np.random.rand(shape_emb[0], shape_emb[1]) * 12 - 6
                        self.embeddings.append(emb)

                    elif self.load_embeddings:
                        # Take the embeddings of the last frame of the sequence
                        last_frame_number = int(last_frame.split("_")[1])
                        # Rise error if embedding is not found
                        assert not df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number].empty, 'Embedding not found for frame: ' + str(last_frame_number)+ '. Check the csv file.' + str(df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number, df_embeddings.columns[1:]].to_numpy())
                        emb = df_embeddings.loc[df_embeddings.iloc[:, 0] == last_frame_number, df_embeddings.columns[1:]].to_numpy()
                        self.embeddings.append(emb)
                        
                if inter_seq: 
                    sts += window # Sliding window of 1 frame between sequences   
                else:      
                    sts = end
                    if inference: 
                        sts = sts - 4 # To overlap sequences while inference for smooth predictions (4 frames) 
                    

    def __len__(self):
        return len(self.sequences)
    
    # Create a function to load the data in parallel
    def load_data(self, frame_path, AEM_path, sal_map_path):
    
        assert os.path.exists(frame_path), 'Image frame has not been found in path: ' + frame_path
        img_frame = cv2.imread(frame_path)
        if img_frame.shape[1] != self.resolution_frames[1] or img_frame.shape[0] != self.resolution_frames[0]:
            img_frame = cv2.resize(img_frame, (self.resolution_frames[1], self.resolution_frames[0]),
                                    interpolation=cv2.INTER_AREA)
        img_frame = img_frame.astype(np.float32)
        img_frame = img_frame / 255.0

        img_frame = torch.FloatTensor(img_frame)
        img_frame = img_frame.permute(2, 0, 1).unsqueeze(0)

        if self.load_OF:
            frame_name = os.path.basename(frame_path)
            flow_map_path = os.path.join(self.path_flow, frame_name.split("_")[0], frame_name)
            assert os.path.exists(flow_map_path), 'Flow map has not been found in path: ' + flow_map_path

            flow_img = cv2.imread(flow_map_path)
            if flow_img.shape[1] != self.resolution_frames[1] or flow_img.shape[0] != self.resolution_frames[0]:
                flow_img = cv2.resize(flow_img, (self.resolution_frames[1], self.resolution_frames[0]), interpolation=cv2.INTER_AREA)
            flow_img = flow_img.astype(np.float32)
            flow_img = flow_img / 255.0

            flow_img = torch.FloatTensor(flow_img)
            flow_img = flow_img.permute(2, 0, 1).unsqueeze(0)
            img_frame = torch.cat((img_frame, flow_img), dim=1)


        if AEM_path is not None:
            assert os.path.exists(AEM_path), 'Image frame has not been found in path: ' + AEM_path
            # Read as bn image
            AEM = cv2.imread(AEM_path, cv2.IMREAD_GRAYSCALE)
            if AEM.shape[1] != self.resolution_AEM[1] or AEM.shape[0] != self.resolution_AEM[0]:
                AEM = cv2.resize(AEM, (self.resolution_AEM[1], self.resolution_AEM[0]),
                                        interpolation=cv2.INTER_AREA)
            AEM = AEM.astype(np.float32)
            AEM = AEM / 255.0

            AEM = torch.FloatTensor(AEM)
            AEM = AEM.unsqueeze(2)
            AEM = AEM.permute(2, 0, 1).unsqueeze(0)
        else:
            AEM = None

        if sal_map_path is not None:
            assert os.path.exists(sal_map_path), 'Saliency map has not been found in path: ' + sal_map_path

            saliency_img = cv2.imread(sal_map_path, cv2.IMREAD_GRAYSCALE)
            if saliency_img.shape[1] != self.resolution_frames[1] or saliency_img.shape[0] != self.resolution_frames[0]:  
                saliency_img = cv2.resize(saliency_img, (self.resolution_frames[1], self.resolution_frames[0]),
                                            interpolation=cv2.INTER_AREA)

            saliency_img = saliency_img.astype(np.float32)
            saliency_img = (saliency_img - np.min(saliency_img)) / (np.max(saliency_img) - np.min(saliency_img))
            saliency_img = torch.FloatTensor(saliency_img).unsqueeze(0).unsqueeze(0)
        else:
            saliency_img = None
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]

        return img_frame, AEM, saliency_img, frame_name

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # total_time_init = time.time()

        frame_img = []
        label = []
        frame_names = []
        AEM_map = []

        if self.loadlogMel:
            # Load the logMel map
            logMel_path = os.path.join(self.path_logMel, self.logMel[idx].split('_')[0], self.logMel[idx])
            assert os.path.exists(logMel_path), 'Image frame has not been found in path: ' + logMel_path
            # Read as bn image and resize
            logMel_img = cv2.imread(logMel_path, cv2.IMREAD_GRAYSCALE)
            if logMel_img.shape[1] != self.resolution_logMel[1] or logMel_img.shape[0] != self.resolution_logMel[0]:
                logMel_img = cv2.resize(logMel_img, (self.resolution_logMel[1], self.resolution_logMel[0]),
                                        interpolation=cv2.INTER_AREA)
            
            logMel_img = logMel_img.astype(np.float32)
            logMel_img = logMel_img / 255.0

            logMel_img = torch.FloatTensor(logMel_img)
            logMel_img = logMel_img.unsqueeze(2)
            logMel_img = logMel_img.permute(2, 0, 1)
            
        if self.loadaudio:
            # Read the audio file
            # Dont show warning messages at loading the audio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_wave, _ = lr.load(os.path.join(self.path_audio, self.audio[idx].split("_")[0],self.audio[idx]), sr=None, mono=True)
            audio_wave = torch.FloatTensor(audio_wave)

            if audio_wave.shape[0] > 128000:
                audio_wave = audio_wave[:128000]
            elif audio_wave.shape[0] < 128000:
                # Replicate the last sample
                # print('WARNING: Audio file has less than 128000 samples. Replicating the last sample to match the length of 128000 samples.')
                audio_wave = torch.cat((audio_wave, audio_wave[-1].repeat(128000 - audio_wave.shape[0])))
            audio_wave = audio_wave.unsqueeze(0) # Add a dimension to the tensor
        
        if self.load_embeddings:
            # Load the embeddings
            embeddings = torch.FloatTensor(self.embeddings[idx])
            embeddings = embeddings.unsqueeze(0) # Add a dimension to the tensor

        frames_paths = [os.path.join(self.path_frames, frame_name.split("_")[0], frame_name) for frame_name in self.sequences[idx]]
        frame_names = [os.path.splitext(os.path.basename(frame_name))[0] for frame_name in self.sequences[idx]]
        
        if self.path_sal_maps is None:
        # Create an array of None to pass to the function
            gt_paths = [None] * len(frames_paths)
        else:
            gt_paths = [os.path.join(self.path_sal_maps, frame_name.split("_")[0], frame_name) for frame_name in self.sequences[idx]]

        if len(self.aems) <= 0:
        # Create an array of None to pass to the function
            aem_paths = [None] * len(frames_paths)
        else:
            aem_paths = [os.path.join(self.path_AEM, aem_frame.split("_")[0], aem_frame) for aem_frame in self.aems[idx]]
            # aem_paths = [os.path.join(self.path_AEM, frame_name.split("_")[0], aem_frame) for aem_frame, frame_name in zip(self.aems[idx],self.sequences[idx])]

        assert len(frames_paths) == len(aem_paths), "Number of frames and AEM maps must be the same."

        if not self.threading:
            for frame_path, AEM_path, gt_path in zip(frames_paths, aem_paths, gt_paths):
                frame, AEM, saliency, frame_name = self.load_data(frame_path, AEM_path, gt_path)
                frame_img.append(frame)
                AEM_map.append(AEM)
                label.append(saliency)
                frame_names.append(frame_name)
        else:

            # Create a thread pool of 20 threads to load the data
            executor = ThreadPoolExecutor(max_workers=20)
            results = executor.map(self.load_data, frames_paths, aem_paths, gt_paths)
            # When all the threads are done, save the results in the corresponding lists
            tmp = []
            for result in results:
                frame_img.append(result[0])
                AEM_map.append(result[1])
                label.append(result[2])
                tmp.append(result[3])

            # Raise error if tmp and frame_names are not the same
            assert tmp == frame_names, "Frame names are not the same."
        
        sample = []
        
        if self.load_frames:
            sample.append(torch.cat(frame_img, 0)) # Frames shape: (batch_size, seq_len, channels, height, width)
        if self.loadaudio:
            sample.append(audio_wave) # Audio shape: (batch_size, channels=1, n_samples)
        if self.loadlogMel:
            sample.append(logMel_img) # LogMel shape: (batch_size, channels=1, height, width)
        if self.loadAEM:
            sample.append(torch.cat(AEM_map, 0)) # AEM shape: (batch_size, seq_len, channels=1, height, width)
        if self.load_embeddings:
            sample.append(embeddings) # Embeddings shape: (batch_size, seq=1, n_features)
        if self.path_sal_maps is not None:
            sample.append(torch.cat(label, 0)) # Ground truth shape: (batch_size, seq_len, channels=1, height, width)
        if self.load_names:
            sample.append(frame_names) # Frame names shape: (batch_size, seq_len)
        # print('Time to load the data: ', time.time() - init_time, 'Pack:', idx)
        return sample