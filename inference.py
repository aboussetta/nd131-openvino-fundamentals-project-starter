#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

        

    def load_model(self, model, device="CPU", cpu_extension=None, request_id=0):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

            
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        # Initialize the plugin
        self.plugin = IECore()
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        # Add a CPU extension, if applicable
        if not(is_supported_layers(self.plugin, self.network)):
            self.plugin.add_extension(cpu_extension, device)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, request_id)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return
        

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    
    def exec_net(self, request_id, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id, inputs={self.input_blob:image})      
        return

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output_blob]

    
def is_supported_layers(plugin, network):
    ### TODO check if all layers are supported
    ### return True if all supported, False otherwise
    layers_map = plugin.query_network(network=network, device_name="CPU")
    layers = network.layers.keys()

    supported = True
    for l in layers:
        if l not in layers_map:
            supported = False
            break

    return supported
        
            
