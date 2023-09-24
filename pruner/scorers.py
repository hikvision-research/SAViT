# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
contains different importance metrics for pruning
"""
def prepare_pruning_list(model, logger, pruning_config):
    pruning_parameters_list = list()
    pruning_layers = pruning_config.pruning_layers
    for module_name, m in model.named_modules():
        if hasattr(m, "do_not_update"):
            if pruning_layers == 0 and 'attn_gate' not in module_name:
                continue
            if pruning_layers == 1 and 'hidden_gate' not in module_name:
                continue
            if pruning_layers == 2 and 'res_gate' not in module_name:
                continue
            m.register_forward_hook(forward_hook)
            m.register_backward_hook(backward_hook)
            pruning_parameters_list.append(m)

    return pruning_parameters_list

def forward_hook(self, input, output):
    self.output = output.detach()

def backward_hook(self, grad_input, grad_output):
    self.grad = (grad_output[0].detach(),)

def taylor1Scorer(params, w_fun=lambda a: -a):
    """
    taylor1Scorer, method 6 from ICLR2017 paper 
    <Pruning convolutional neural networks for resource efficient transfer learning>
    """
    if len(params.grad[0].shape) == 4:
        score = (params.grad[0] * params.output).abs().mean(-1).mean(
                                -1).mean(0)
    else:
        score = (params.grad[0] * params.output).abs().mean(1).mean(0)  

    return score, 0

def taylor2Scorer(params):
    """
    method 22 from CVPR2019 paper, best in their paper
    <Importance estimation for neural network pruning>
    """
    if len(params.grad[0].shape) == 4:
        nunits = params.grad[0].shape[1]
    else:
        nunits = params.grad[0].shape[-1]
    # score = (params.weight * params.weight.grad).data.pow(2).view(nunits, -1).sum(1)
    score = (params.weight * params.weight.grad).data.pow(2)

    # the two calculation is identical
    # score = (params.weight * params.weight.grad).data.pow(2)
    # equal to calculation belows:
    # if type(params)==list:
    #     score = torch.zeros_like(params[0].grad[0])
    #     N = len(params)
    #     for i in range(N):
    #         score+=(params[i].grad[0]*params[i].output)
    #     score = score.sum(1).sum(0).data.pow(2)
    return score, 0

def taylor3Scorer(params):
    """
    method 23 from CVPR2019 paper, full grad
    <Importance estimation for neural network pruning>
    """
    if len(params.grad[0].shape) == 4:
        full_grad = (params.grad[0] * params.output).sum(-1).sum(-1)
    else:
        full_grad = (params.grad[0] * params.output).sum(1)

    score = full_grad.data.pow(2).sum(0)

    return score, 0

def taylor4Scorer(params, w_fun=lambda a: -a):
    """
    method 1 from 2019 NeuIPS paper
    <Are Sixteen Heads Really Better than One>
    """
    if len(params.grad[0].shape) == 4:
        taylor_im = (params.grad[0] * params.output).sum(-1).abs()
        if params.mask is not None:
            taylor_im = taylor_im[params.mask]
        score = taylor_im.sum(-1).sum(0)
        denom = taylor_im.size(0) * taylor_im.size(2)
    else:
        taylor_im = (params.grad[0] * params.output).abs()
        score = taylor_im.sum(1).sum(0)
        denom = taylor_im.size(0) * taylor_im.size(1)    
    return score, denom