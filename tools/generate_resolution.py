from PIL import Image
import argparse
import os
import cv2
from pathlib import Path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--scale',type=int, default=2)
    args = parser.parse_args()

    source_folder = os.path.join(args.path,'images_8/')
    dest_folder = os.path.join(args.path,'images_'+str(args.scale))

    Path(dest_folder).mkdir(exist_ok = True)
    
    for img_name in os.listdir(source_folder):
        # print(img_name)
        img = cv2.imread(source_folder+img_name)
        h,w,c = img.shape
        resized = cv2.resize(img, (w//args.scale,h//args.scale), interpolation=cv2.INTER_AREA)
        dest_folder_name = os.path.join(dest_folder,img_name)
        # print(dest_folder)
        cv2.imwrite(dest_folder_name, resized)