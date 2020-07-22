#!/usr/bin/env python
# encoding: utf-8
#
# Copyright Nvidia Corporation
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


class Layer:

    def __init__(self, name, src_layers):
        '''

        Base class for a layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer
        '''
        self.name = name
        self.src_layers = src_layers

    def get_name(self):
        return self.name

    def get_src_layers(self):
        return self.src_layers

    def get_src_layers_count(self):

        if self.src_layers is None:
            return 0
        elif isinstance(self.src_layers, list):
            return len(self.src_layers)
        else:
            return 1

    def get_parameters(self):
        params = dict()
        params['name'] = self.name
        params['top'] = self.name

        if self.get_src_layers_count() > 1:
            s = []
            for layer in self.get_src_layers():
                s.append(layer.get_name())
            params['bottom'] = s
        elif self.get_src_layers_count() == 1:
            params['bottom'] = self.get_src_layers().get_name()

        return {k: v for k, v in params.items() if v is not None}

    def __str__(self):
        return str(self.get_parameters())


class Dropout(Layer):

    def __init__(self, name, src_layers, rate=0.2):
        '''

        Dropout layer
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer
        :param rate: float
            Specifies the ratio to be dropped out.
        '''
        super().__init__(name, src_layers)
        self.rate = rate

    def get_parameters(self):
        d_params = super().get_parameters()
        d_params['type'] = 'Dropout'
        d_params['rate'] = self.rate
        return {k: v for k, v in d_params.items() if v is not None}


class FullyConnected(Layer):

    def __init__(self, name, src_layers, n=1024):
        '''
        Fully connected layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer
        :param n: int
            Specifies the number of neurons in the layer.
        '''
        super().__init__(name, src_layers)
        self.n = n

    def get_parameters(self):
        f_params = super().get_parameters()
        f_params['type'] = 'InnerProduct'
        f_params['fc_param'] = { "num_output" : self.n }
        return {k: v for k, v in f_params.items() if v is not None}
