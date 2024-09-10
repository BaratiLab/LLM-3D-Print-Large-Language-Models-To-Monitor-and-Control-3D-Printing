import os
import base64
import imutils
import requests
import glob2
import json
import pandas as pd
import cv2
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def resize_image(img_path, factor, gray=False, save_path="./temp"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(img_path)
    resized = imutils.resize(img, width = img.shape[1]//factor, inter=cv2.INTER_AREA)

    if gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"{save_path}/{img_path.split('/')[-1]}", resized)
    return f"{save_path}/{img_path.split('/')[-1]}"

def encode_image(image_path):
    """encodes image for chatgpt"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def send_image(image_path, resize_factor=2, system_prompt="", user_prompt="", previous_image_path=None):
    base64_image = encode_image(resize_image(img_path=image_path, factor=resize_factor))
    chat = ChatOpenAI(model = "gpt-4o")

    if previous_image_path is not None:
        base_64_image_previous = encode_image(resize_image(img_path=previous_image_path, factor=resize_factor))

        output = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": f"This is the image of previous segment. Defects are already identified and an attempt to solve was made for this segment. Donot consider this part of the print to identify the defects. Identify the defects only present in the current part of the print."},
                    {"type": "image_url", 
                    "image_url": {
                            "url": f"data:image/jpeg;base64,{base_64_image_previous}",
                            "detail": "auto"
                    },
                    },

                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", 
                    "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                    },
                    },

                ]
            )
        ]
    )
        return output
    else:
        output = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", 
                    "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                    },
                    },
                ]
            )
        ]
    )
        return output


def send_image_to_openai(image_path,system_prompt,user_prompt, resize_factor,api_key):
    """Sends an image to OpenAI's API and prints the response."""
    
    # Encode the image
    base64_image = encode_image(resize_image(img_path=image_path, factor=resize_factor))
    # base64_image_previous = encode_image(resize_image(img_path=image_path_2, factor=resize_factor))

    # Set the headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {   
                "role": "user",
                "content": f"{user_prompt}\n![image](data:image/jpeg;base64,{base64_image})"
            }
        ],
    }


    # Send the POST request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Print the JSON response from the server
    print(response.json())

    return response.json()["choices"][0]["message"]["content"], response
