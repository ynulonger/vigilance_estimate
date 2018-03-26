import scipy.io as sio
import numpy as np
import pandas as pd
import os

dir = "./perclos_labels/"
filelist = os.listdir(dir)
print(filelist)
for file in filelist:

        labels = sio.loadmat(dir+file)["perclos"]
        labels = np.squeeze(labels)
        print(labels.shape)
        Labels = pd.DataFrame({"labels:":labels})
        writer = pd.ExcelWriter(
                "./result/excels/" +file+ ".xlsx")
        Labels.to_excel(writer, 'result', index=True)
        writer.save()