from chain import *
import json
from utils import *
import requests
import os
import json
import time
import cv2
from loguru import logger
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from image_inference import *
import glob2

from PIL import Image, ImageDraw, ImageFont
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
url = "172.24.115.43" #change URL HERE 
OPENAI_API_KEY = "sk-" #ADD KEY HERE

os.environ["QT_QPA_PLATFORM"] = "offscreen"

def get_toolhead_state(url,debug=False):
    """
    Fetches the state of the toolhead from a Moonraker 3D printer.

    Args:
    url (str): The base URL to the Moonraker API.
    """
    full_url = f"http://{url}/printer/objects/query?toolhead"
    try:
        # Send a request to the Moonraker API to get the state of the toolhead
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response and extract toolhead status
        data = response.json()
        toolhead_state = data.get('result', {}).get('status', {}).get('toolhead', {})

        # Print the state of the toolhead
        if debug:
            print("Toolhead State:")
            for key, value in toolhead_state.items():
                print(f"{key}: {value}")

        return True, toolhead_state

    except requests.RequestException as e:
        print(f"Failed to retrieve toolhead state: {e}")

        return False, e
    
def crop_combine(top_loc, front_loc, save_loc):
    load_top = np.array(Image.open(top_loc))
    load_front = np.array(Image.open(front_loc))
    # plt.imshow(load_top[200:,600:1600], origin='lower')
    # plt.show()
    # plt.imshow(load_front[:,200:1250])
    # plt.show()

    # Crop the images
    load_top_cropped = load_top[200:, 650:1550]
    load_front_cropped = load_front[200:, 400:1250]

    # Determine the new image size
    height_top, width_top, _ = load_top_cropped.shape
    height_front, width_front, _ = load_front_cropped.shape

    # Create a new image with combined width
    combined_image = np.zeros((max(height_top, height_front), width_top + width_front, 3), dtype=np.uint8)

    # Place the images side by side
    combined_image[:height_top, :width_top, :] = load_top_cropped
    combined_image[:height_front, width_top:width_top + width_front, :] = load_front_cropped

    # Convert to PIL image for drawing
    combined_image_pil = Image.fromarray(combined_image)
    draw = ImageDraw.Draw(combined_image_pil)

    font = ImageFont.load_default(size=100)

    draw.text((10, 10), "Top", font=font, fill=(255, 255, 255))
    draw.text((width_top + 10, 10), "Front", font=font, fill=(255, 255, 255))

    combined_image_with_text = np.array(combined_image_pil)

    # Display the combined image
    plt.imshow(combined_image_with_text)
    plt.tight_layout()
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_loc, bbox_inches='tight', pad_inches=0)
    plt.close()



def resume_print(ip_address):
    """
    Pause the current print job on a 3D printer managed by MainsailOS.

    Args:
    ip_address (str): The IP address of the MainsailOS server.
    """
    url = f"http://{ip_address}/printer/print/resume"
    headers = {'Content-Type': 'application/json'}
    data = {}  # Depending on your setup, this might need to be 'M25' or another specific command

    # try:
    response = requests.post(url, json=data, headers=headers),



def get_printer_state(url):
    """
    Fetches the state of the printer from a Moonraker 3D printer.
    
    Args:
    url (str): The base URL to the Moonraker API.
    """
    url = f"http://{url}/printer/objects/query?print_stats"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data

def get_timelapse_image(frame_no, path):
    """
    Fetches a timelapse image from a Moonraker 3D printer.
    
    Args:
    frame_no (int): The frame number of the timelapse image to fetch.
    """
    url = f"http://172.24.115.43/server/files/timelapse_frames/frame{frame_no}.jpg"
    response = requests.get(url)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)


def get_image(url, path, mode="front"):
    """gets the image of the printer encodes it"""
    print("Getting front image")
    if mode=="front":
        url = f"http://{url}/webcam3/?action=snapshot"
    else:
        url = f"http://{url}/webcam/?action=snapshot"
    response = requests.get(url)
    if response.status_code == 200:
        image_path = path
        with open(image_path, "wb") as f:
            f.write(response.content)
        # print("Image saved as printer_image.jpg")
    else:
        print("Failed to retrieve image, status code:", response.status_code)



def check_previous_images(path):
    img_list = glob2.glob(path+"/top_**.jpg")

    num=[0]
    for i in img_list:
        num.append(int(i.split("_")[-1].split(".")[0]))

    return max(num)


def runner(printer_url="172.24.115.43",
           image_save_path="./results_3/cmu/layer_images",
           save_path="./results_3/cmu",
           image_resize_factor=2,
           openai_api_key=OPENAI_API_KEY,
           ):

    flag= True

    # prompts = json.load(open("./prompts/system_prompt.json", "r"))
    image_system_prompt=load_text_file("./prompts/image_system_prompts.txt")
    image_user_prompt=load_text_file("./prompts/image_user_prompt.txt")
    

    while flag:

        max_tries = 5
        print(f"\033[91mChecking printer state\033[0m")

        current_state = get_printer_state(url)
        # print(current_state)
            
        printer_status = current_state["result"]["status"]["print_stats"]["state"]
        current_layer = current_state["result"]["status"]["print_stats"]["info"]["current_layer"]

        print(f"\033[91mPrinter Status: {printer_status}\033[0m")
        print(f"\033[91mCurrent Layer: {current_layer}\033[0m")

        if printer_status == "printing":
            time.sleep(20)

        if printer_status == "complete":
            print("\033[91mPrinting Complete. Exiting.\033[0m")
            flag=False

        
        if printer_status == "paused":
            print("\033[91mPrinter is paused. Take snapshot.\033[0m")
            time.sleep(10)

            # nn = check_previous_images(image_save_path)
            nn = current_layer
            # nn=nn+1
            pre=2
            if nn>=3:
                pre=nn-1

            # print(nn)
        
            # image_top = get_timelapse_image(str("%06d" % nn), image_save_path+f"/top_layer_{nn}.jpg")
            image_front = get_image(printer_url, image_save_path+ f"/front_layer_{nn}.jpg")
            image_top = get_image(printer_url, image_save_path+ f"/top_layer_{nn}.jpg", mode="top")

            print("Image saved.")
            time.sleep(5)

            # croped = crop_combine(top_loc=image_save_path+ f"/top_layer_{nn}.jpg", front_loc=image_save_path+ f"/front_layer_{nn}.jpg")
        
            # prompts = json.load(open("./prompts/system_prompt.json", "r"))
            image_system_prompt=load_text_file("./prompts/image_system_prompts.txt")
            image_user_prompt=load_text_file("./prompts/image_user_prompt.txt")

            # failures,r = send_image(image_save_path+f"/top_layer_{nn}.jpg",system_prompt=prompts["system_prompt_eyes"],user_prompt=f"This the current image at layer {current_layer}. Identify the most visually prominent defects in the current layer.", resize_factor=image_resize_factor, api_key=openai_api_key, image_path_2=image_save_path+f"/front_layer_{pre}.jpg")

            if nn==3:

                croped = crop_combine(top_loc=image_save_path+ f"/top_layer_{nn}.jpg", front_loc=image_save_path+ f"/front_layer_{nn}.jpg", save_loc=image_save_path+ f"/combined_{nn}.jpg")

                failures = send_image(image_save_path+f"/combined_{nn}.jpg", system_prompt=image_system_prompt, user_prompt = image_user_prompt, resize_factor=2)
                # print("skipping correction for first layer")
                # resume_print(url)

            else:
                print("Running image inference")
                croped = crop_combine(top_loc=image_save_path+ f"/top_layer_{nn}.jpg", front_loc=image_save_path+ f"/front_layer_{nn}.jpg", save_loc=image_save_path+ f"/combined_{nn}.jpg")
                failures = send_image(image_save_path+f"/combined_{nn}.jpg", system_prompt=image_system_prompt, user_prompt = image_user_prompt, resize_factor=2, previous_image_path=image_save_path+f"/combined_{pre}.jpg")

            #save failures to a file
            with open(save_path+f"/failures_{nn}.txt", "w") as f:
                f.write(failures.content)

            print(f"\033[92m Detected Failures:\n {failures}\033[0m")

            #check if a file exists
            if os.path.exists(save_path+f"/previous_solution_{pre}.txt"):
                previous_solution = load_text_file(save_path+f"/previous_solution_{pre}.txt")
            else:
                previous_solution = []

            graph = get_graph()

            print("\033[94mRunning LLM AGENT")

            logfile = save_path+f"/log{nn}.log"
            
            logger.add(logfile, colorize=True, enqueue=True)
            handler_1 = FileCallbackHandler(logfile)
            handler_2 = StdOutCallbackHandler()
            
            reasoning_planner=load_text_file("./prompts/info_reasoning.txt")
            # observation=load_text_file("failure.txt")
            observation = failures.content  
            printer_objects = load_text_file("./prompts/printer_objects.txt")
            solution_reasoning= load_text_file("./prompts/solution_reasoning.txt")
            gcode_cmd = load_text_file("./prompts/gcode_commands.txt")

            out = graph.invoke(
                {
                    "internal_messages": ["Given the failure (if any) plan for what information is required to identify the issue, query printer for the required information, plan the solution steps to solve the problem, execute the solution plan, resume print and finish.\n If no problem is detected, resume print."],

                    "printer_url": printer_url,

                    "information_known":["Printer url is http://172.24.115.43","Filament type is PLA","printer model is creatlity ender 5 plus", f"Printer status is paused", "Current layer is {current_layer}", "Tool position is at the home position", "BED is perfectly Calibrated", "Nozzle diameter is 1mm", "There are no issues with the nozzle and printer hardware.", "Layer height is 0.3", "Infill pattern is aligned rectiliner"],
                    "observations": observation,
                    "reasoning": reasoning_planner,
                    "solution_reasoning": solution_reasoning, 
                    "printer_obj": printer_objects,
                    "adapted_recon_reasoning" : [],
                    "adapter_solution_reasoning" : [],
                    "gcode_commands":gcode_cmd,
                    "previous_solution" : previous_solution,
                    "solution_steps":[],


                },
                {"callbacks":[handler_1, handler_2]},
                debug=True
                
            )

            previous_solution = out["solution_steps"]
            with open(save_path+f"/previous_solution_{nn}.txt", "w") as f:
                f.write(str(previous_solution))

            with open(save_path+f"/LLM_out_{nn}.txt", "w") as f:
                f.write(str(out))

                

                
runner()

# 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
