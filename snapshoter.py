import requests
import os
import re
import json
import asyncio
import time
import cv2
import glob2
url = "172.24.115.43"

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
    response = requests.post(url, json=data, headers=headers)
        # print(response.json())
        # print(response.text)
    #     if response.status_code == 200:
    #         # print("Print Resumed successfully")
    #     else:
    #         print(f"Resumed: {response.status_code} - {response.text}")
    # except Exception as e:
    #     print(f"Error sending pause command: {e}")

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
    url = f"http://172.26.79.51/server/files/timelapse_frames/frame{frame_no}.jpg"
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

# while True:
#     print(f"Checking printer state")

#     current_state = get_printer_state(url)
#     # print(current_state)
    
#     printer_status = current_state["result"]["status"]["print_stats"]["state"]
#     current_layer = current_state["result"]["status"]["print_stats"]["info"]["current_layer"]
#     total_layers = current_state["result"]["status"]["print_stats"]["info"]["total_layer"]


#     _,toolhead = get_toolhead_state(url,debug=False)
#     print(f"Printer status: {printer_status}, Current Layer: {current_layer}")

#     if printer_status == "printing":
#         time.sleep(10)

#     if printer_status == "paused":
#         print("Printer is paused. Take snapshot.")
#         time.sleep(2)

#         nn = check_previous_images("./images/square_test_default")
#         nn=nn+1
#         print(nn)
    
#         # image_top = get_timelapse_image(str("%06d" % nn), f"./images/square_test_default/top_layer_{nn}.jpg")
#         image_front = get_image("172.26.79.51", f"./results_2/multi_bishop/no_llm/front_layer_{nn}.jpg")
#         print("Image saved.")
#         image_top = get_image("172.26.79.51",  f"./results_2/multi_bishop/no_llm/top_layer_{nn}.jpg", mode="top")
#         time.sleep(5)
#         resume_print(url)

#     if printer_status != "printing":
#         if printer_status != "paused":
#             if current_layer == total_layers:
#                 print("Print completed.")
#                 break
    
"""""""""""""""""
for normal stuff uncomment the below code
"""""""""""""""""
while True:
    print(f"Checking printer state")

    current_state = get_printer_state(url)
    # print(current_state)
    
    printer_status = current_state["result"]["status"]["print_stats"]["state"]
    current_layer = current_state["result"]["status"]["print_stats"]["info"]["current_layer"]
    total_layers = current_state["result"]["status"]["print_stats"]["info"]["total_layer"]


    _,toolhead = get_toolhead_state(url,debug=False)
    print(f"Printer status: {printer_status}, Current Layer: {current_layer}")

    if printer_status == "printing":
        time.sleep(5)

    if printer_status == "paused":
        print("Printer is paused. Take snapshot.")
        time.sleep(2)
        if current_layer ==0:
            # image_top = get_timelapse_image(str("%06d" % 1), f"./results_2/multi_bishop/no_llm/top_layer_{1}.jpg")
            image_top = get_image("172.24.115.43", f"./results_2/puzzle_piece/no_llm/top_layer_{1}.jpg", mode="top")
            image_front = get_image("172.24.115.43", f"./results_2/puzzle_piece/no_llm/front_layer_{1}.jpg")
        else:
            # image_top = get_timelapse_image(str("%06d" % current_layer), f"./results_2/multi_bishop/no_llm/top_layer_{current_layer}.jpg")
            image_top = get_image("172.24.115.43", f"./results_2/puzzle_piece/no_llm/top_layer_{current_layer}.jpg", mode="top")
            image_front = get_image("172.24.115.43", f"./results_2/puzzle_piece/no_llm/front_layer_{current_layer}.jpg")
        print("Image saved.")
        time.sleep(1)
        resume_print(url)

    if printer_status != "printing":
        if current_layer == total_layers:
            print("Print completed.")
            break


# http://172.24.115.43/