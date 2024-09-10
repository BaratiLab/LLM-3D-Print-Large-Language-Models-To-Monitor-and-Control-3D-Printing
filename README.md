# LLM-3D-Print-Large-Language-Models-To-Monitor-and-Control-3D-Printing

### TO DO: Make user friendly and easy to use. Code Updated Regularly !

Website: [https://sites.google.com/andrew.cmu.edu/printerchat](https://sites.google.com/andrew.cmu.edu/printerchat)

## Getting Started

### Printer
1. Use an tool such as [KIAUH](https://github.com/dw-0/kiauh) to compile and
install Klipper firmware.
    - Make sure to install Crowsnest plugin to enable camera streaming.

2. Update config files accordingly, sample configs are provided for reference.


### Code Changes
1. Get API key for chatgpt-4o from OpenAPI
2. Set up the environment from env.yml
3. Change Printer URL in runner.py, snapshooter.py and tools.py
4. In ./prompts/image_user_prompt.txt (point 6) give the description of the object.
5. Run as python runner.py


### Information:

1. All prompts in ./prompts
2. Results and run log for single layer prints in results.zip
3. Results and run log for multilayer prints as presented in the paper are in results_2.zip
4. Result and run log for video in results_3.zip
5. Multi Layer print Gcode in Gcode file.
