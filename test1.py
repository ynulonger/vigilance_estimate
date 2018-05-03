import scipy.io as sio
import sys
data_dir = "./EEG_Feature_5Bands"
filename = sys.argv[1]
data = sio.loadmat(data_dir +"/"+ filename)
print(data.keys())