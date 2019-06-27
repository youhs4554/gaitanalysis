BATCH_SIZE_OF_TFMODEL = 1
MODEL_PATH = '/media/hossay/hdd1/GaitData/pretrained/C3D/conv3d_deepnetA_sport1m_iter_1900000_TF.model'
MEAN_FILE = 'train01_16_128_171_mean.npy'
FRAME_HOME = "/media/hossay/hdd1/GaitData/CroppedFrameArrays"
FEATS_SAVE_DIR = "/media/hossay/hdd1/GaitData/EncodedFeatures"
FRAMES_PER_CLIP = 16
FRAME_MAXLEN=300
FEATS_MAXLEN=20
target_columns = [
                   'Velocity', 
                  'Cadence',
                  'Functional Amb. Profile',
                 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
                  'Stride Length(cm)/L', 'Stride Length(cm)/R',
                  'HH Base Support(cm)/L', 'HH Base Support(cm)/R',
                 'Swing Time(sec)/L', 'Swing Time(sec)/R',
                 'Stance Time(sec)/L', 'Stance Time(sec)/R',
                 'Double Supp. Time(sec)/L', 'Double Supp. Time(sec)/R',
                  #'Toe In / Out/L', 'Toe In / Out/R'
                  ]