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
            Specifies source layer(s) for a layer.
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
            Specifies source layer(s) for a layer.
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
            Specifies source layer(s) for a layer.
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


class ELU(Layer):

    def __init__(self, name, src_layers, alpha=1.0):
        '''
        ELU layer
       :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        :param alpha: float
            Specifies alpha parameter for the layer.
        '''
        super().__init__(name, src_layers)
        self.alpha = alpha

    def get_parameters(self):
        e_params = super().get_parameters()
        e_params['type'] = "ELU"
        e_params['elu_param'] = { "elu_param" : self.alpha }
        return {k: v for k, v in e_params.items() if v is not None}


class Reshape(Layer):

    def __init__(self, name, src_layers, leading_dim):
        '''
        Reshape layer
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        :param leading_dim: int
            Specifies the leading dimension in the rehape layer.
        '''

        super().__init__(name, src_layers)
        self.leading_dim = leading_dim

    def get_parameters(self):
        r_params = super().get_parameters()
        r_params['type'] = "Reshape"
        r_params['leading_dim'] = self.leading_dim
        return {k: v for k, v in r_params.items() if v is not None}


class Concat(Layer):

    def __init__(self, name, src_layers):
        '''
        Concat layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        '''
        super().__init__(name, src_layers)

    def get_parameters(self):
        c_params = super().get_parameters()
        c_params['type'] = 'Concat'


class Slice(Layer):

    def __init__(self, name, src_layers, ranges):
        '''
        Slice layer.
        *** Note that slice layer is a bit complicated. It can output multiple instances.
        Therefore, I will be treating it differently ***
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        '''
        super().__init__(name, src_layers)
        self.ranges = ranges

    def get_parameters(self):
        s_params = super().get_parameters()

        # change top if there are multiple ranges
        length = len(self.ranges)
        if length > 1:
            s = []
            for i in range(0, length):
                s.append(self.get_name()+"_"+i)
            s_params['top'] = s

        s_params['ranges'] = self.ranges
        s_params['type'] = 'Slice'
        return {k: v for k, v in s_params.items() if v is not None}


class DistributedSlotSparseEmbeddingHash(Layer):

    def __init__(self, name, src_layers, vocabulary_size, load_factor, embedding_vec_size, combiner):
        '''
        Distributed Slot Sparse Embedding Hash layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        :param vocabulary_size: int or long
            Specifies the maximum vocabulary size for the embedding.
        :param load_factor: float
            Specifies the radio of the loaded vocabulary to capacity of the hashtable.
        :param embedding_vec_size: int
            Specifies the vector size of an embedding weight (value). Then the memory used in this hashtable will be
            vocabulary_size x embedding_vec_size / load_factor
        :param combiner: Boolean
            If set to 0 then sum, if set to 1 then mean
        '''
        super().__init__(name, src_layers)
        self.vocabulary_size = vocabulary_size
        self.load_factor = load_factor
        self.embedding_vec_size = embedding_vec_size
        self.combiner = combiner

    def get_parameters(self):
        d_params = super().get_parameters()
        d_params['type'] = 'DistributedSlotSparseEmbeddingHash'
        d_params['sparse_embedding_hparam'] = { "vocabulary_size": self.vocabulary_size,
                                                "load_factor": self.load_factor,
                                                "embedding_vec_size": self.embedding_vec_size,
                                                "combiner": self.combiner}
        return {k: v for k, v in d_params.items() if v is not None}


class RELU(Layer):
    def __init__(self, name, src_layers):
        '''
        RELU layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        '''
        super().__init__(name, src_layers)

    def get_parameters(self):
        r_params = super().get_parameters()
        r_params['type'] = 'ReLu'
        return {k: v for k, v in r_params.items() if v is not None}


class BinaryCrossEntropyLoss(Layer):
    def __init__(self, name, src_layers):
        '''
        Binary cross entropy loss layer.
        :param name: str
            Specifies name of a layer.
        :param src_layers: Layer or a list of Layers
            Specifies source layer(s) for a layer.
        '''
        super().__init__(name, src_layers)

    def get_parameters(self):
        b_params = super().get_parameters()
        b_params['type'] = 'BinaryCrossEntropyLoss'
        return {k: v for k, v in b_params.items() if v is not None}


class Dense:

    def __init__(self, name, dim=0):
        '''
        Dense model (features)
        :param name: str
            Specifies name.
        :param dim: int
            Specifies dense dimension
        '''
        self.name = name
        self.dim = dim

    def get_parameters(self):
        d_params = dict()
        d_params['top'] = self.name
        d_params['dense_dim'] = self.dim
        return {k: v for k, v in d_params.items() if v is not None}

    def get_name(self):
        return self.name

class Sparse:

    def __init__(self, name, slot_num, max_feature_num_per_sample=100):
        '''

        :param name: str
            Specifies name.
        :param slot_num: int
            Specifies slot number.
        :param max_feature_num_per_sample: int
            Specifies maximum feature number per sample.
        '''
        self.name = name
        self.slot_num = slot_num
        self.max_feature_num_per_sample = max_feature_num_per_sample

    def get_parameters(self):
        s_params = dict()
        s_params['top'] = self.name
        s_params['type'] = "DistributedSlot"
        s_params['max_faeture_num_per_sample'] = self.max_feature_num_per_sample
        s_params['slot_num'] = self.slot_num
        return {k: v for k, v in s_params.items() if v is not None}

    def get_name(self):
        return self.name

class Label:

    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def get_parameters(self):
        l_params = dict()
        l_params['top'] = self.name
        l_params['label_dim'] = self.dim
        return {k: v for k, v in l_params.items() if v is not None}

    def get_name(self):
        return self.name

class Data(Layer):

    def __init__(self, name, label, dense, sparse, source=None, eval_source=None, check='Sum'):
        '''
        Data layer
        :param name: str
            Specifies name of layer.
        :param label: Label
            Specifies label of model
        :param dense: Dense
            Specifies dense features
        :param sparse: list of Sparse
            Specifies sparse features (embeddings)
        :param source: str
            Specifies source file for training data.
        :param eval_source: str
            Specifies source file for test data.
        :param check: str
            Specifies if check.
        '''
        super().__init__(name=name,src_layers=None)
        self.label = label
        self.dense = dense
        if isinstance(sparse, list):
            self.sparse = sparse
        else:
            self.sparse = [sparse]
        self.source = source
        self.eval_source = eval_source
        self.check = check

    def get_parameters(self):
        d_params = super().get_parameters()
        del d_params['top']
        d_params['type'] = 'Data'
        d_params['source'] = self.source
        d_params['eval_source'] = self.eval_source
        d_params['check'] = self.check
        d_params['label'] = {'label': self.label.get_parameters()}
        d_params['dense'] = {'dense': self.dense.get_parameters()}
        s=[]
        for sp in self.sparse:
            s.append(sp.get_parameters())
        d_params['sparse'] = {'sparse': s}
        return {k: v for k, v in d_params.items() if v is not None}

    def __str__(self):
        return str(self.get_parameters())