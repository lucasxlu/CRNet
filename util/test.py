import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd

# df = pd.read_csv('./cvsplit/jaffe.csv', sep=' ')
# files = [df['PIC'][i] + str(df['#'][i]) + '' for i in range(len(df['#'].tolist()))]
# for index, row in df.iterrows():
#     if index > 0:
#         print(np.argmax(np.array(row[1: 7].tolist())))
#
# labels = []


for _ in os.listdir('E:\DataSet\Face\jaffe'):
    im = Image.open(os.path.join('E:\DataSet\Face\jaffe', _))
    print("Generating jpeg for %s" % _)
    out = im.convert("RGB")
    out.save(os.path.join('E:\DataSet\Face\jaffeJPG', _.replace('.tiff', '.jpg')), "JPEG", quality=90)
