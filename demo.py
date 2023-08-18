import gradio as gr
import subprocess
import os
import threading
import cv2
import natsort
import time
import pickle
import copy
# os.chdir('..')
# os.chdir('cyclegan1')
# path = os.getcwd()
# print(path)
# os.chdir('..')
def mmdetection(image):
    try:
        print(os.getcwd())
        os.chdir("mmdetection")

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'data/coco/demo_images/{len(os.listdir("data/coco/demo_images"))+1}.jpg',image)

        subprocess.call(["python", "tools/test.py"])

        output_dir = 'output_demo/vis'
        output_list = natsort.natsorted(os.listdir(output_dir))
        output = output_dir +'/' + output_list[-1]

        output = cv2.imread(output)
        output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)

        pkl_output_dir = 'pkl_output'
        pkl_output_list = natsort.natsorted(os.listdir(pkl_output_dir))
        pkl_ouput = pkl_output_dir + '/' + pkl_output_list[-1]

        with open(pkl_ouput,'rb') as f:
            text_output = pickle.load(f)
        temp = list(text_output.keys())

        os.chdir('..')
        return output,temp
    except: 
        os.chdir("/home/he0/deep_daiv.spring")
        return None, None

    

def cyclegan(text):
    try:
        current_dir = os.getcwd()
        os.chdir('mmdetection')

        pkl_dir = 'pkl_output'
        pkl_list = natsort.natsorted(os.listdir(pkl_dir))
        pkl_ouput = pkl_dir + '/' + pkl_list[-1]
        with open(pkl_ouput,'rb') as f:
            text_output = pickle.load(f)
        seg = text_output[text]
        result_image_name = pkl_list[-1].split('.')[0] + '_' + text + '_' + seg[0] + '.jpg'

        image = cv2.imread('results_demo/'+result_image_name)
        img = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        os.chdir('..')


        cv2.imwrite(current_dir + '/cyclegan1/datasets/segmented/testA/' + pkl_list[-1].split('.')[0] + '.jpg',img)
        os.chdir('cyclegan1')
        subprocess.call(["python",'test.py' ,'--dataroot', 'datasets/segmented', '--name', 'seg_to_hanzi', '--model', 'cycle_gan' ])
        
        image_dir = 'results/seg_to_hanzi/test_latest/images'
        image_list = natsort.natsorted(os.listdir(image_dir))
        image_1 = cv2.imread(image_dir + '/' + image_list[-1])
        iamge_2 = cv2.imread(image_dir + '/' + image_list[-2])
        
        os.chdir('..')

        return image, image_1, iamge_2
    except:
        os.chdir("/home/he0/deep_daiv.spring")
        return None, None, None
    

with gr.Blocks() as demo:
    # gr.Markdown("choose segmented iamge to generate hanza image")
    with gr.Row():
        with gr.Tab("Preprocessing using RTMdet"):
            input_1 = gr.Image(type='numpy',label='image')
            button_1 = gr.Button("Run")
            output_1 = [gr.Image(type='numpy'), gr.Text()]
            print(output_1)
    with gr.Row():
        with gr.Tab("Generate Hanza data using CycleGan"):
            input_2 = gr.Text()
            button_2 = gr.Button("Run")
            output_2 = [gr.Image(type='numpy'), gr.Image(type='numpy'),gr.Image(type = 'numpy')]

    button_1.click(fn=mmdetection, inputs=input_1, outputs=output_1)
    button_2.click(fn =cyclegan, inputs=input_2, outputs=output_2)

demo.launch(share=True)

# preprocess.launch(share = True)


