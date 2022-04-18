import cv2
import numpy as np
import os
import glob
import time
from itertools import groupby
from utils import focus_stack_pyramid

if __name__=='__main__':
    
    input_dir = "/home/hassan/Work/02_FISH/01_Codes/fish-repo/01_Stacking Module/Input"

    images_list = glob.glob(input_dir+"/*.png")+glob.glob(input_dir+"/*.jpg")
    files_list = [{os.path.dirname(x): os.path.basename(x)} for x in images_list]
    res = [list(i) for j, i in groupby(files_list, lambda a: list(a.values())[0].split('_')[0])]

    time_list = []
    for stack_files in res:
        image_name = list(stack_files[0].values())[0].split('_')[0]
        input_path = list(stack_files[0].keys())[0]
        out_path = "/home/hassan/Work/02_FISH/01_Codes/fish-repo/01_Stacking Module/Input"
        out_dir = os.path.join(out_path, input_path.split('/')[-2])
        stack_files = [os.path.join(input_path, list(x.values())[0]) for x in stack_files]
        try:
            os.mkdir(out_dir)
        except:
            pass
        t1 = time.time()
        out_img = focus_stack_pyramid(stack_files)
        t2 = time.time()
        time_list.append(t2 - t1)
        out_image_path = os.path.join(out_dir,"{}.jpg".format(image_name))
        cv2.imwrite(out_image_path,out_img)
    print(np.mean(time_list))