'''
Code adapted from the original implementation by Erwan DAVID (IPI, LS2N, Nantes, France), 2018
E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (...). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
'''
import numpy as np
from metrics import AUC_Judd, AUC_Borji, NSS, CC, SIM, KLD, EMD, normalize, MSE, MAE
import os
import cv2
import pandas as pd
import config as cfg
import tqdm
EPSILON = np.finfo('float').eps

METRICS = {
    # Check table 4 of "What Do Different Evaluation Metrics Tell Us about Saliency Models?" by Bylinskii et al.
    "AUC_Judd": [AUC_Judd, False, 'fix'], # Binary fixation map
    "AUC_Borji": [AUC_Borji, False, 'fix'], # Binary fixation map
    "NSS": [NSS, False, 'fix'], # Binary fixation map
    "CC": [CC, True, 'sal'], # Saliency map
    "SIM": [SIM, True, 'sal'], # Saliency map
    "EMD": [EMD, False, 'sal'], # Saliency map
    "KLD": [KLD, False, 'sal'], # Saliency map
    "MSE": [MSE, True, 'sal'], # Saliency map
    "MAE": [MAE, True, 'sal'] } # Saliency map

def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None):
    values = []

    for metric in KEYS_ORDER:

        func = METRICS[metric][0]
        sim = METRICS[metric][1]
        compType = METRICS[metric][2]

        if not sim:
            if compType == "fix" and fixmap1 is not None and fixmap2 is not None:
                m = (func(salmap1, fixmap2)\
                   + func(salmap2, fixmap1))/2
            elif compType == "fix" and fixmap2 is not None:
                m = func(salmap1, fixmap2)
            elif compType == "sal":
                m = (func(salmap1, salmap2)\
                   + func(salmap2, salmap1))/2
            else:
                print("Error: No fixation map provided for fixation map comparison")
        else:
            m = func(salmap1, salmap2)
        values.append(m)
    return values

def uniformSphereSampling(N):
    gr = (1 + np.sqrt(5))/2
    ga = 2 * np.pi * (1 - 1/gr)

    ix = iy = np.arange(N)

    lat = np.arccos(1 - 2*ix/(N-1))
    lon = iy * ga
    lon %= 2*np.pi

    return np.concatenate([lat[:, None], lon[:, None]], axis=1)


if __name__ == "__main__":
     # ======================== Parameters ========================
    PRED_SM_PATH = cfg.predicted_salmaps_path
    GT_SM_PATH = cfg.gt_salmap_path
    GT_FIX_PATH = cfg.gt_fixations_file_path
    OUTPUT_PATH = cfg.output_cvs_file_path
    WIDTH, HEIGHT = cfg.salmaps_resolution[0], cfg.salmaps_resolution[1]
    KEYS_ORDER = cfg.metrics_to_compute
    SAMPLING_TYPE = cfg.sampling_type
    # =============================================================
    
    # ================= Weighting pre-computation =================
    print("SAMPLING_TYPE: ", SAMPLING_TYPE)

    if SAMPLING_TYPE.split("_")[0] == "Sphere":
        print(int(SAMPLING_TYPE.split("_")[1]))
        unifS = uniformSphereSampling( int(SAMPLING_TYPE.split("_")[1]))
        unifS[:, 0] = unifS[:, 0] / np.pi * (HEIGHT-1)
        unifS[:, 1] = unifS[:, 1] / (2*np.pi) * (WIDTH-1)
        unifS = unifS.astype(int)
    elif SAMPLING_TYPE == "Sin":
        VerticalWeighting = np.sin(np.linspace(0, np.pi, HEIGHT)) # latitude weighting
        # plt.plot(np.arange(height), VerticalWeighting);plt.show()
    # =============================================================

    # Create dataframe to store metrics
    # Add video and frame columns to dataframe
    column_names = ['Video', 'Frame']
    column_names.extend(KEYS_ORDER)
    metrics_df = pd.DataFrame(columns=column_names)

    videos = os.listdir(PRED_SM_PATH)

    for video in tqdm.tqdm(videos):
        # Compute metrics for each frame
        frames = os.listdir(os.path.join(PRED_SM_PATH,video))
        frames = sorted(frames, key=lambda x: int(x.split("_")[1].split('.')[0]))
        # Load fixation maps cvs
        if GT_FIX_PATH is not None:
            df_video = pd.read_csv(os.path.join(GT_FIX_PATH, video + ".csv"))

        for frame in frames:

            #============= Read saliency and fixation maps =============#
            # Load saliency maps. Read images as grayscale
            pred_salmap = cv2.imread(os.path.join(PRED_SM_PATH, video, frame),cv2.IMREAD_GRAYSCALE)
            if not os.path.exists(os.path.join(GT_SM_PATH, video, frame)):
                print('Not salmap in:', os.path.join(GT_SM_PATH, video, frame))
                continue
            gt_salmap = cv2.imread(os.path.join(GT_SM_PATH, video, frame),cv2.IMREAD_GRAYSCALE)
            # If saliency map pred is all zeros, skip frame
            if np.all(pred_salmap == 0):
                continue
            pred_salmap = normalize(pred_salmap, method='sum')
            gt_salmap = normalize(gt_salmap, method='sum')

            # Take the fixation map corresponding to the frame
            if GT_FIX_PATH is not None:
                df_frame = df_video.loc[df_video['frame'] == int(frame.split("_")[1].split('.')[0])]
                gt_fix = np.zeros((HEIGHT, WIDTH))
                img_coords = np.mod(np.round(df_frame[['u','v']].values * np.array((WIDTH, HEIGHT))), np.array((WIDTH, HEIGHT))-1.0).astype(int)
                # Set fixation points to 1
                gt_fix[img_coords[:, 1], img_coords[:, 0]] = 1

            # Rise a warning if the ground-truth saliency map is not the same size as the parameters
            if gt_salmap.shape != (HEIGHT, WIDTH):
                print("Warning: ground-truth saliency maps does not have the indicated size WxH = {}.".format((WIDTH, HEIGHT)))
            # Resize pred_salmap to gt_salmap size
            pred_salmap = cv2.resize(pred_salmap, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            # ============================================================#

            #============= Apply sampling method =============#
            # Apply uniform sphere sampling if specified
            if SAMPLING_TYPE.split("_")[0] == "Sphere":
                pred_salmap = pred_salmap[unifS[:, 0], unifS[:, 1]]
                gt_salmap = gt_salmap[unifS[:, 0], unifS[:, 1]]
                if GT_FIX_PATH is not None:
                    gt_fix = gt_fix[unifS[:, 0], unifS[:, 1]]
            # Weight saliency maps vertically if specified
            elif SAMPLING_TYPE == "Sin":
                pred_salmap = pred_salmap * VerticalWeighting[:, None] + EPSILON
                gt_salmap = gt_salmap * VerticalWeighting[:, None] + EPSILON

            pred_salmap = normalize(pred_salmap, method='sum')
            gt_salmap = normalize(gt_salmap, method='sum')
            # ================================================#

            #============= Compute and store metrics =============#
            # Compute similarity metrics
            if GT_FIX_PATH is not None:
                values = getSimVal(pred_salmap, gt_salmap, fixmap2=gt_fix)
            else:
                values = getSimVal(pred_salmap, gt_salmap)

            #Print metrics
            # print("Frame {} of video {} processed".format(frame, video))
            # print("Similarity metrics: ", values)
    
            # Add video, frame and metrics values to dataframe row
            row = [video, frame]
            row.extend(values)
            metrics_df = pd.concat([metrics_df, pd.DataFrame([row], columns=column_names)])
            # ====================================================#

        # Save the results to a csv file
        metrics_df.to_csv(OUTPUT_PATH, index=False)