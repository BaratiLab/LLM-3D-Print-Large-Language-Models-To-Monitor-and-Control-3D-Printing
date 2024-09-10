from langchain.tools import tool
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException
import json
import time

@tool
def resume_print(printer_url:str)->str:
    """
    Pause the current print job on a 3D printer managed by MainsailOS.

    Args:
    ip_address (str): The IP address of the MainsailOS server.
    """
    #change printer URL here
    printer_url = "http://172.24.115.43"
    url = f"{printer_url}/printer/print/resume"
    headers = {'Content-Type': 'application/json'}
    data = {}  # Depending on your setup, this might need to be 'M25' or another specific command

    # try:
    response = requests.post(url, json=data, headers=headers)
    return "Print job resumed."

@tool
def query_printer(object_name:str) -> dict:
    """
    Query the Moonraker API to get state of object_name and return the response data.

    Args:
        input: object_name.

    Returns:
        dict: The data returned by the Moonraker API.
    """
    # url = f"{printer_url}/printer/objects/query?{object_name}"
    url = f"http://172.24.115.43/printer/objects/query?{object_name}"
    # Check if the request was successful
    try:
        response = requests.get(url)
        # response.raise_for_status()
        out = response.json()
        print("Output ", str(out))
        return out
    except requests.exceptions.HTTPError as e:
        return {"error": str(e)}

@tool
def query_gCode(gcode_command:str) -> dict:
    """
    Query the Moonraker API to get current retraction rate.

    Args:
        input: Dict containing the printer_url and object_name.

    Returns:
        dict: The data returned by the Moonraker API.
    """
    endpoint = f"http://172.24.115.43/printer/gcode/script"

    # Data payload for the POST request
    data = {
        "script": gcode_command
    }
    # Check if the request was successful
    response = requests.post(endpoint, json=data)
    # print(response.json())
    time.sleep(3)
    get_url=f"http://172.24.115.43/server/gcode_store?count=1"
    out = requests.get(get_url)
    out=out.json()
    # print(out)

    return {"message": str(out)}


@tool
def change_parameters(gcode_command:str)->dict:
    """
    Changes the parameters of the printer by executing G-code scripts.

    Args:
        input: gcode_command.

    Returns:
    dict: 
        A dictionary containing the response status and message.
    """
    # Define the endpoint for executing G-code scripts
    endpoint = f"http://172.24.115.43/printer/gcode/script"

    # Data payload for the POST request
    data = {
        "script": gcode_command
    }

        # Send the POST request to Moonraker API
    response = requests.post(endpoint, json=data)
    time.sleep(3)

    get_url=f"http://172.24.115.43/server/gcode_store?count=1"
    out = requests.get(get_url)
    out=out.json()

    return {"message": str(out)}