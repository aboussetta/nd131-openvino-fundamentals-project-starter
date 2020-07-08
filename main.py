#!/usr/bin/env python3


"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension, request_id=0)
    net_input_shape = infer_network.get_input_shape()
    print("net_input_shape: {}".format(net_input_shape))
    
    ### TODO: Handle the input stream ###
    
        # Checks for live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    # Get and open video capture
    try:
        cap = cv2.VideoCapture(input_validated)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    cap.open(input_validated)
    flag, frame = cap.read()

    print("video opened")
    # Grab the shape of the input 
    w = int(cap.get(3))
    h = int(cap.get(4))
    
    #print(f"weight: {w}, heigh: {h}")
    #print(f"{w} is an {h} company.")
    print ("weigh: {}, heigh: {}".format(w,h))

    #Define and iniatilize some working variables
    request_id=0

    ### TODO: Loop until stream is over ###
    #Define and iniatilize some working variables
    timing = 0
    counter = 0
    previous_counter = 0
    current_count = 0
    total_count = 0
    previous_duration = 0

    # Process frames until the video ends, or process is exited
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        print ("reading video")

        # Waits for a pressed key.
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
        print ("net_input_shape[3]: {}, net_input_shape[2]: {}".format(net_input_shape[3], net_input_shape[2]))

        i_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = i_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        
        ### TODO: Start asynchronous inference for specified request ###
        # Perform inference on the frame
        duration= 0
    
        infer_network.exec_net(request_id, p_frame)
        print ("start inference video")

        ### TODO: Wait for the result ###
        # Get the output of inference
        if infer_network.wait(request_id=0) == 0:

            print ("Get the results of inference")

            ### TODO: Get the results of the inference request ###
            result = plugin.extract_output(request_id)
            
            print ("current_count: {}, total_count: {}, duration : {}".format(current_count,total_count,duration ))

            ### TODO: Extract any desired stats from the results ###
            current_count, total_count, duration = get_stats(result, p_frame, prob_threshold, w, h, counter, previous_counter, total_count, current_count, duration, previous_duration, timing)
            
            print ("current_count: {}, total_count: {}, duration : {}".format(current_count,total_count,duration ))

            current_count, total_count, duration
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": current_count, "total": total_count}))
            client.publish("person/duration", json.dumps({"duration": duration}))


        ### TODO: Send the frame to the FFMPEG server ###
        print("here")
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()


        ### TODO: Write an output image if `single_image_mode` ###
        # Write out the frame
        if single_image_mode == True:
            out.write(frame)
        
    cap.release()
    cv2.destroyAllWindows()

    return
            
    def get_stats(result, frame, prob_threshold, width, heigh, counter, previous_counter, total_count, current_count, duration, previous_duration, timing):
        # Define a variable
        pointer = 0
        probs = result[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > prob_threshold:
                pointer += 1
                box = result[0, 0, i, 3:]
                p1 = (int(box[0] * width), int(box[1] * heigh))
                p2 = (int(box[2] * width), int(box[3] * heigh))
                frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        
        if pointer != counter:
            previous_counter = counter
            counter = pointer
            if timing >= 3:
                previous_duration = timing
                timing = 0
            else:
                timing = previous_duration + timing
                previous_duration = 0  # unknown, not needed in this case
        else:
            timing += 1
            if timing >= 3:
                current_count = counter
                if timing == 3 and current_count > previous_counter:
                    total_count += current_count - previous_counter
                elif timing == 3 and current_count < previous_counter:
                    duration = int((previous_duration / 10.0) * 1000)

        return current_count, total_count, duration


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Set log to INFO
    log.basicConfig(level=log.CRITICAL)

    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
