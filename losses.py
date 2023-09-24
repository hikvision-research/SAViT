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
Implementation of the knowledge distillation loss.
distill the pruned model using unpruned model and cnn as teachers.
"""
import torch
from torch.nn import functional as F
from functools import partial

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are fed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    loss_batch = torch.sum(- targets_prob * student_likelihood, dim=-1)
    return loss_batch.mean()

def cal_relation_loss(student_attn_list, teacher_attn_list, Ar=1):
    '''
    distilling relation between input images
    '''
    layer_num = len(student_attn_list)
    relation_loss = 0.
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        B, N, Cs = student_att[0].shape
        _, _, Ct = teacher_att[0].shape
        for i in range(3):
            for j in range(3):
                # (B, Ar, N, Cs // Ar) @ (B, Ar, Cs // Ar, N)
                # (B, Ar) + (N, N)
                matrix_i = student_att[i].view(B, N, Ar, Cs//Ar).transpose(1, 2) / (Cs/Ar)**0.5
                matrix_j = student_att[j].view(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)
                As_ij = (matrix_i @ matrix_j) 

                matrix_i = teacher_att[i].view(B, N, Ar, Ct//Ar).transpose(1, 2) / (Ct/Ar)**0.5
                matrix_j = teacher_att[j].view(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)
                At_ij = (matrix_i @ matrix_j)
                relation_loss += soft_cross_entropy(As_ij, At_ij)
    return relation_loss/(9. * layer_num)

def cal_hidden_loss(student_hidden_list, teacher_hidden_list):
    '''
    distilling mlp features
    '''
    layer_num = len(student_hidden_list)
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        hidden_loss +=  torch.nn.MSELoss()(student_hidden, teacher_hidden)
    return hidden_loss/layer_num

def cal_hidden_relation_loss(student_hidden_list, teacher_hidden_list):
    '''
    distilling relation between mlp features
    '''
    layer_num = len(student_hidden_list)
    B, N, Cs = student_hidden_list[0].shape
    _, _, Ct = teacher_hidden_list[0].shape
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        student_hidden = torch.nn.functional.normalize(student_hidden, dim=-1)
        teacher_hidden = torch.nn.functional.normalize(teacher_hidden, dim=-1)
        student_relation = student_hidden @ student_hidden.transpose(-1, -2)
        teacher_relation = teacher_hidden @ teacher_hidden.transpose(-1, -2)
        hidden_loss += torch.mean((student_relation - teacher_relation)**2) * 49 #Window size x Window size
    return hidden_loss/layer_num

def soft_distillation(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd, T=1):
    '''
    soft distillation loss
    '''
    distillation_loss_full_cls = F.kl_div(
        F.log_softmax(outputs / T, dim=1),
        F.log_softmax(teacher_outputs / T, dim=1),
        reduction='sum',
        log_target=True
    ) * (T * T) / outputs.numel()
    distillation_loss_full_dist = F.kl_div(
        F.log_softmax(outputs_kd / T, dim=1),
        F.log_softmax(teacher_outputs_kd / T, dim=1),
        reduction='sum',
        log_target=True
    ) * (T * T) / outputs_kd.numel()
    return distillation_loss_full_cls + distillation_loss_full_dist

def hard_distillation(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd):
    '''
    hard distillation loss
    '''
    distillation_loss_full_cls = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))
    distillation_loss_full_dist = F.cross_entropy(outputs_kd, teacher_outputs_kd.argmax(dim=1))
    return distillation_loss_full_cls + distillation_loss_full_dist

def mse(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd):
    '''
    mse distillation loss
    '''
    distillation_loss_full_cls = F.mse_loss(outputs, teacher_outputs)
    distillation_loss_full_dist = F.mse_loss(outputs_kd, teacher_outputs_kd)
    return distillation_loss_full_cls + distillation_loss_full_dist

class PrunedDistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, gt: str,
                 teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float,
                 teacher_model_full,
                 distillation_type_full, alpha_full, tau_full,
                 distillation_attn_full_im, distillation_alpha_attn_full_im, 
                 distillation_mlp_full_im, distillation_alpha_mlp_full_im,
                 distillation_type_full_im, alpha_full_im):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.gt = gt
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        
        self.teacher_model_full = teacher_model_full
        self.distillation_type_full = distillation_type_full
        self.alpha_full = alpha_full
        self.tau_full = tau_full
        self.distillation_attn_full_im = distillation_attn_full_im
        self.distillation_alpha_attn_full_im = distillation_alpha_attn_full_im
        self.distillation_mlp_full_im = distillation_mlp_full_im
        self.distillation_alpha_mlp_full_im = distillation_alpha_mlp_full_im

        self.distillation_type_full_im = distillation_type_full_im
        self.alpha_full_im = alpha_full_im
        if self.distillation_type_full_im == 'none':
            self.forward = self.forward_wo_im
        else:
            self.forward = self.forward_with_im_minivit
        self.choose_loss()

    def choose_loss(self):
        if self.distillation_type_full == 'soft':
            self.dist_full_loss = partial(soft_distillation, T=self.tau_full)
        elif self.distillation_type_full == 'hard':
            self.dist_full_loss = hard_distillation
        elif self.distillation_type_full == 'mse':
            self.dist_full_loss = mse
        if self.distillation_type_full_im == 'mse':
            self.dist_full_mlp_loss = cal_hidden_loss
        elif self.distillation_type_full_im == 'rel':
            self.dist_full_mlp_loss = cal_hidden_relation_loss

    def forward_wo_im(self, inputs, outputs, labels):
        '''
        only distill the logits, without intermediate features.
        '''
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)

        if not self.gt:
            base_loss = torch.zeros_like(base_loss)

        if self.distillation_type == 'none':
            loss_cnn = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
            loss_cnn = self.alpha * distillation_loss

        if self.distillation_type_full == 'none':
            loss_full = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs, teacher_outputs_kd = self.teacher_model_full(inputs)
            loss_full = self.alpha_full * self.dist_full_loss(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd)
        return base_loss + loss_cnn + loss_full

    def forward_with_im_minivit(self, inputs, outputs, labels):
        '''
        only distill the logits and intermediate features.
        refer to minivit paper <MiniViT: Compressing Vision Transformers with Weight Multiplexing>.
        '''
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd, outputs_attns, outputs_mlps = outputs
        base_loss = self.base_criterion(outputs, labels)

        if not self.gt:
            base_loss = torch.zeros_like(base_loss)

        if self.distillation_type == 'none':
            loss_cnn = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
            loss_cnn = self.alpha * distillation_loss

        if self.distillation_type_full == 'none':
            loss_full = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs, teacher_outputs_kd, \
                t_outputs_attns, t_outputs_mlps = self.teacher_model_full(inputs)
            
            loss_full = self.alpha_full * self.dist_full_loss(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd)

            loss_full_im = torch.zeros_like(base_loss)
            if self.distillation_type_full_im != 'none':
                if self.distillation_attn_full_im:
                    loss_full_im += self.distillation_alpha_attn_full_im * cal_relation_loss(outputs_attns, t_outputs_attns)
                if self.distillation_mlp_full_im:
                    loss_full_im += self.distillation_alpha_mlp_full_im * self.dist_full_mlp_loss(outputs_mlps, t_outputs_mlps)
                loss_full_im = self.alpha_full_im * loss_full_im
        return base_loss + loss_cnn + loss_full + loss_full_im

    def forward_with_im_tinybert(self, inputs, outputs, labels):
        '''
        only distill the logits and intermediate features.
        refer to tinybert paper <TinyBERT: Distilling BERT for Natural Language Understanding>.
        '''
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd, outputs_attns, outputs_mlps = outputs
        base_loss = self.base_criterion(outputs, labels)

        if not self.gt:
            base_loss = torch.zeros_like(base_loss)

        if self.distillation_type == 'none':
            loss_cnn = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
            loss_cnn = self.alpha * distillation_loss

        if self.distillation_type_full == 'none':
            loss_full = torch.zeros_like(base_loss)
        else:
            with torch.no_grad():
                teacher_outputs, teacher_outputs_kd, \
                teach_full_outputs_attns, teach_full_outputs_mlps = self.teacher_model_full(inputs)
            
            loss_full = self.alpha_full * self.dist_full_loss(outputs, teacher_outputs, outputs_kd, teacher_outputs_kd)

            loss_full_im = torch.zeros_like(base_loss)
            if self.distillation_type_full_im == 'mse':
                for outputs_attn, teach_full_outputs_attn in zip(outputs_attns, teach_full_outputs_attns):
                    loss_full_im += F.mse_loss(outputs_attn, teach_full_outputs_attn)
                for outputs_mlp, teach_full_outputs_mlp in zip(outputs_mlps, teach_full_outputs_mlps):
                    loss_full_im += F.mse_loss(outputs_mlp, teach_full_outputs_mlp)
                loss_full_im = self.alpha_full_im * loss_full_im

        return base_loss + loss_cnn + loss_full + loss_full_im
