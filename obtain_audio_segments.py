import os
import librosa as lr
import soundfile as sf
import tqdm
import cv2
import numpy as np

def compute_log_mel(sequence):

    mel_signal = lr.feature.melspectrogram(y=sequence, sr=sr)
    power_to_db = lr.power_to_db(mel_signal, ref=np.max)
    # Up to here you could plot it with librosa.specshow()
    # Next steps turn it into an image
    power_to_db = power_to_db + np.abs(np.min(power_to_db))
    power_to_db = np.round((power_to_db / np.max(power_to_db)) * 256)
    power_to_db = power_to_db.astype(np.uint8)
    power_to_db = power_to_db[::-1,::]
    spec = cv2.applyColorMap(power_to_db,cv2.COLORMAP_MAGMA)
    return spec


PATH_TO_FRAMES = './data/DSAV360/frames'
PATH_TO_AUDIOS = './data/DSAV360/audio_files'
OUT_AUDIO_SEGMENTS = './data/DSAV360/audio_segments'
OUT_LOGMEL = './data/DSAV360/logMel_images'
FRAMES_TEMPORAL_WINDOW = 20
FPS = 60
EXTRACT_LOGMEL = False


videos = os.listdir(PATH_TO_FRAMES)
# If it does not exist, create the folder to save the audio segments
if not os.path.exists(OUT_AUDIO_SEGMENTS):
    os.makedirs(OUT_AUDIO_SEGMENTS)

# If it does not exist, create the folder to save the logMel images
if EXTRACT_LOGMEL:
    if not os.path.exists(OUT_LOGMEL):
        os.makedirs(OUT_LOGMEL)

for video in tqdm.tqdm(videos):
    # Serach for the audio file correspondent to the video
    audio_path = os.path.join(PATH_TO_AUDIOS, video + '.wav')
    if not os.path.exists(audio_path):
        print("Audio file not found for video", video)
        continue
    # Read the audio file
    audio, sr = lr.load(audio_path, sr=None, mono=True)

    # Read the correspondent frames
    frames = os.listdir(os.path.join(PATH_TO_FRAMES, video))
    # Sort the frames by name 
    frames = sorted(frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    time_between_frames = 1 / FPS

    # Create the folder to save the audio segments
    if not os.path.exists(os.path.join(OUT_AUDIO_SEGMENTS, video)):
        os.makedirs(os.path.join(OUT_AUDIO_SEGMENTS, video))

    if EXTRACT_LOGMEL:
        if not os.path.exists(os.path.join(OUT_LOGMEL, video)):
            os.makedirs(os.path.join(OUT_LOGMEL, video))

    # Save the audio segment for each frame as the audio between the current frame and the previous one
    nf = FRAMES_TEMPORAL_WINDOW
    for frame in frames[FRAMES_TEMPORAL_WINDOW:]:
        # Compute the init and end time of the audio segment
        init_time = int(frames[nf-FRAMES_TEMPORAL_WINDOW].split('_')[-1].split('.')[0]) / FPS
        end_time = int(frame.split('_')[-1].split('.')[0]) / FPS
        # Save the audio segment
        audio_segment = audio[int(sr * init_time):int(sr * end_time)]
        sf.write(os.path.join(OUT_AUDIO_SEGMENTS, video, frame.split('.')[0] + '.wav'), audio_segment, sr)
        nf += 1

        if EXTRACT_LOGMEL:
            logMel_image = compute_log_mel(audio_segment)
            cv2.imwrite(os.path.join(OUT_LOGMEL, video, frame.split('.')[0] + '.png'), logMel_image)
