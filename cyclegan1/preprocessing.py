from pathlib import Path
import os
import shutil
import cv2
import numpy as np

def file_del():

    for file in Path(root_dir).iterdir():
        print(file)
        for image in Path(file).iterdir():
            num = int(str(image).split('/')[3].split('.')[0])
            print(num)
            if num == 1:
                continue
            else:
                if os.path.isfile(str(image)):
                    os.remove(str(image))

def file_extract():

    for i,file in enumerate(Path(root_dir).iterdir()):
        src = str(file)
        shutil.move(str(file), root_dir +'/'+ str(i) + '.png')

def canny_operator(image, min_threshold, max_threshold, color_type=None):

    result = cv2.Canny(image, min_threshold, max_threshold)
    image_flipped = np.where(result == 0, 255, np.where(result == 255, 0, result))
    return image_flipped

def morphology(image,kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.dilate(image,kernel,iterations=1)
    return eroded

if __name__ == '__main__':
    root_dir = 'datasets/set1/trainB'
    # file_extract()
    dir = 'datasets/segmented/trainA'
    files = os.listdir(dir)
    i = 0
    img = '000000474039_person_1.jpg'
    image = cv2.imread(img,cv2.IMREAD_COLOR)
    image_canny = canny_operator(image,150,250)
    cv2.imwrite('asdf.jpg',image_canny)




   