from settings import DIRECTORY_PATH, TRAIN_FOLDER, BALANCED_PATH, RGB_BALANCED, PREPROCESSED_BALANCED, RGB_PREPROCESSED_BALANCED
from imageio.v2 import imread, imsave
from settings import  RGB_DIRECTORY_PATH, CLAHE_PATH,RGB_CLAHE_PATH
from PIL import Image
#f_name = '/home/atlas/datasets/rostami/balancedpneumothorax_rgb/' + TRAIN_FOLDER + "/normal/6684_train_0_.png"




import os
source_dir = CLAHE_PATH
dest_dir  = RGB_CLAHE_PATH

from PIL import Image

for folder in ['train', 'test', 'validation']:
    for imclass in ['normal','pnemothorax']:
            localdir = source_dir+folder+'/'+imclass+'/'
            destdir = dest_dir+folder+'/'+imclass+'/'
            for r, d, f in os.walk(localdir):
                for file in f:
                    try:
                        img = Image.open(localdir+file).convert('RGB')
                        img.save(destdir+file)
                        img.close()
                    except Exception as ex:
                        print(ex)
                        print(localdir+file)

