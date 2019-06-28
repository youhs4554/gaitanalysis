BATCH_SIZE_OF_TFMODEL = 1
MODEL_PATH = '/data/GaitData/pretrained/C3D/conv3d_deepnetA_sport1m_iter_1900000_TF.model'
MEAN_FILE = 'train01_16_128_171_mean.npy'
FRAME_HOME = "/data/GaitData/CroppedFrameArrays"
FEATS_SAVE_DIR = "/data/GaitData/EncodedFeatures"
FRAMES_PER_CLIP = 16
FRAME_MAXLEN=300
FEATS_MAXLEN=20

'''
# obtained from "Regression experiments.ipynb", which exploit block algorithm

Cluster 01  ['Stride Length(cm)', 'Functional Amb. Profile', 'Velocity']
Cluster 02  ['Cycle Time(sec)', 'Stance Time(sec)', 'Double Supp. Time(sec)']
Cluster 03  ['Swing Time(sec)', 'Cadence', 'HH Base Support(cm)']

'''

target_columns = [
                  # group 1
                  'Stride Length(cm)/L', 'Stride Length(cm)/R',
                  'Functional Amb. Profile',
                   'Velocity', 

                  # group 2
                 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
                 'Stance Time(sec)/L', 'Stance Time(sec)/R',
                 'Double Supp. Time(sec)/L', 'Double Supp. Time(sec)/R',
    
                 # group 3
                 'Swing Time(sec)/L', 'Swing Time(sec)/R',
                 
                 # group 4
                 'Cadence',
    
                 # group 5
                 'HH Base Support(cm)/L', 'HH Base Support(cm)/R',
                  
                ]


