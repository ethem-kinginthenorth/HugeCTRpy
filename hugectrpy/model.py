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


class Model:

    def __init__(self, solver, optimizer, layers):
        '''

        :param solver: Solver
            Specifies a solver for model.
        :param optimizer: Optimizer
            Specifies an optimizer for model.
        :param layers: list of Layers
            Specifies layers for model.
        '''
        self.solver = solver
        self.optimizer = optimizer
        self.layer = layers

    def get_parameters(self):
        parameters = dict()
        parameters['solver'] = self.solver.get_parameters()
        parameters['optimizer'] = self.optimizer.get_parameters()

        return {k: v for k, v in parameters.items() if v is not None}

    def __str__(self):
        import json
        return json.dumps( self.get_parameters())


class Solver:

    def __init__(self, lr_policy='fixed', display=1000, max_iter=30000, gpu=[0],
                 batch_size=512, snapshot=100000, snapshot_prefix="./",
                 eval_interval=1000, eval_batches=60, mixed_precision=None,
                 dense_model_file=None, sparse_model_file=None):
        '''

        :param lr_policy: str, optional
            Specifies the learning rate policy. Only fixed is supported now.
        :param display: int
            Specifies the interval to print out loss.
        :param max_iter: int
            Specifies the maximum number of iterations to be executed in training.
        :param gpu: list of integer
            Specifies GPU indices used in a training process. For example: [[0,1],[2,3]] means that two node are used,
            and in the first node GPUs with index 0 and 1 are used and 2, 3 in the second node.
        :param batch_size: int
            Specifies mini batch size used in training.
        :param snapshot: int
            Specifies intervals to save a checkpoint in a file with the prefix of `snapshot_prefix`.
        :param snapshot_prefix: str
            Specifies the prefix for the checkpoint to be saved.
        :param eval_interval: int
            Specifies intervals of evaluation on test set.
        :param eval_batches: int
            Specifies the number of batches will be used in loss calculation for evaluation. Average loss of those
            batches will be reported.
        :param mixed_precision: int, optional
            If set with an integer, mixed precision will be used and that integer will be used as scaler.
            Supported values are 128, 256, 512, and 1024.
        :param sparse_model_file: list of str, optional
            Specifies model files for sparse model. No need to config if train from scratch)file of sparse models.
            In v2.1 multi-embeddings are supported in one model. Each embedding will have one model file.
        :param dense_model_file:
            Specifies model file for dense model. No need to config if train from scratch
        '''
        self.lr_policy = lr_policy
        self.display = display
        self.max_iter = max_iter

        if isinstance(gpu, int):
            self.gpu = [gpu]
        else:
            self.gpu = gpu

        self.batch_size = batch_size
        self.snapshot = snapshot
        self.snapshot_prefix = snapshot_prefix
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.mixed_precision = mixed_precision
        self.dense_model_file = dense_model_file
        self.sparse_model_file = sparse_model_file

    def get_parameters(self):
        parameter_list = dict()
        parameter_list['lr_policy'] = self.lr_policy
        parameter_list['display'] = self.display
        parameter_list['max_iter'] = self.max_iter
        parameter_list['gpu'] = self.gpu
        parameter_list['batchsize'] = self.batch_size
        parameter_list['snapshot'] = self.snapshot
        parameter_list['snapshot_prefix'] = self.snapshot_prefix
        parameter_list['eval_interval'] = self.eval_interval
        parameter_list['eval_batches'] = self.eval_batches
        parameter_list['mixed_precision'] = self.mixed_precision
        parameter_list['dense_model_file'] = self.dense_model_file
        parameter_list['sparse_model_file'] = self.sparse_model_file

        # this is kind of a sanity check
        return {k: v for k, v in parameter_list.items() if v is not None}

    def __str__(self):
        return str( self.get_parameters())


class Optimizer:

    def __init__(self, global_update=False, lr=0.01):
        '''
        :param global_update: Boolean
            By default optimizer will only update the hot columns of embedding in each iterations.
            If `global_update` set true, optimizer will update all the columns. Note that this option
            slows down the training.
        :param lr: float
            Specifies the learning rate for optimizer.
        '''

        self.global_update = global_update
        self.lr = lr


class AdamOptimizer(Optimizer):

    def __init__(self, gloabl_update=False, lr=0.001, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=0.00000001):
        '''

        Adam optimizer.

        :param gloabl_update: Boolean
            By default optimizer will only update the hot columns of embedding in each iterations.
            If `global_update` set true, optimizer will update all the columns. Note that this option
            slows down the training.
        :param lr_rate: float
            Specifies the learning rate for optimizer.
        :param alpha: float
            Specifies the alpha parameter in adam optimizer.
        :param beta1: float
            Specifies the exponential decay rate for the very first moment for optimizer.
        :param beta2: float
            Specifies the exponential decay rate for the next moment for optimizer
        :param epsilon: float
            Specifies the epsilon parameter in adam optimizer.
        '''

        super().__init__(gloabl_update, lr)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_parameters(self):
        adam_list = dict()
        adam_list['alpha'] = self.alpha
        adam_list['beta1'] = self.beta1
        adam_list['beta2'] = self.beta2
        adam_list['epsilon'] = self.epsilon
        adam_list['learning_rate'] = self.lr

        optimizer_list = dict()
        optimizer_list['type'] = 'Adam'
        optimizer_list['global_update'] = self.global_update
        optimizer_list['adam_hparam'] = {k: v for k, v in adam_list.items() if v is not None}

        return {k: v for k, v in optimizer_list.items() if v is not None}

    def __str__(self):
        return str(self.get_parameters())


class MomentumSGD(Optimizer):

    def __init__(self, global_update=False, lr=0.01, momentum=0.0):
        '''
        
        :param global_update: Boolean
            By default optimizer will only update the hot columns of embedding in each iterations.
            If `global_update` set true, optimizer will update all the columns. Note that this option
            slows down the training.
        :param lr: float
            Specifies the learning rate for optimizer.
        :param momentum:
            Specifies the ratio to be used while updating the weights.
        '''

        super().__init__(global_update, lr)
        self.momentum = momentum

    def get_parameters(self):

        momentum_list = dict()
        momentum_list['momentum_factor'] = self.momentum
        momentum_list['learning_rate'] = self.lr

        optimizer_list = dict()
        optimizer_list['type'] = 'MomentumSGD'
        optimizer_list['global_update'] = self.global_update
        optimizer_list['momentum_sgd_hparam'] = {k: v for k, v in momentum_list.items() if v is not None}

        return {k: v for k, v in optimizer_list.items() if v is not None}

    def __str__(self):
        return str(self.get_parameters())


class Nesterov(Optimizer):

    def __init__(self, global_update=False, lr=0.01, momentum=0.0):
        '''

        :param global_update: Boolean
            By default optimizer will only update the hot columns of embedding in each iterations.
            If `global_update` set true, optimizer will update all the columns. Note that this option
            slows down the training.
        :param lr: float
            Specifies the learning rate for optimizer.
        :param momentum:
            Specifies the ratio to be used while updating the weights.
        '''

        super().__init__(global_update, lr)
        self.momentum = momentum

    def get_parameters(self):
        nesterov_list = dict()
        nesterov_list['momentum_factor'] = self.momentum
        nesterov_list['learning_rate'] = self.lr

        optimizer_list = dict()
        optimizer_list['type'] = 'Nesterov'
        optimizer_list['global_update'] = self.global_update
        optimizer_list['momentum_sgd_hparam'] = {k: v for k, v in nesterov_list.items() if v is not None}

        return {k: v for k, v in optimizer_list.items() if v is not None}

    def __str__(self):
        return str(self.get_parameters())