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
contains important definition of pruning class and pruning function
"""
import os
import copy
import numpy as np
import pickle
import math
import torch
from .scorers import *

def get_model_complexity_info(model, logger, only_total=False, show=True):
    """get the model flops information, including attention only (0), 
       mlp only (1) and whole model (2)
    """
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = model.flops(2)
    if only_total:
        if show:
            logger.info('number of total GFLOPs:{}'.format(total_flops/1e9))
        return 0, 0, total_flops.item()    
    attn_flops = model.flops(0)
    fc_flops = model.flops(1)
    if show:
        logger.info('number of params:{}'.format(n_parameters))
        logger.info('number of Attention GFLOPs:{}'.format(attn_flops/1e9))
        logger.info('number of FC GFLOPs:{}'.format(fc_flops/1e9))
        logger.info('number of total GFLOPs:{}'.format(total_flops/1e9))

    return attn_flops.item(), fc_flops.item(), total_flops.item()

class pruning_config(object):
    """
    pruning config class
    analysing pruning arguments and integrate them into this class.
    """
    def __init__(self, args_dict):
        def initialize_parameter(object_name, key, args, def_value):
            value = def_value
            if key in args.keys():
                value = args[key]
            setattr(object_name, key, value)
        initialize_parameter(self,'pruning_momentum', args_dict,0.9)
        initialize_parameter(self,'pruning_per_iteration', args_dict,100)
        initialize_parameter(self,'start_pruning_after_n_iterations', args_dict,0)
        initialize_parameter(self,'maximum_pruning_iterations', args_dict,10000)
        initialize_parameter(self,'starting_neuron', args_dict,0)
        initialize_parameter(self,'pruning_frequency', args_dict,0)
        initialize_parameter(self,'pruning_iterative', args_dict,False)
        initialize_parameter(self,'pruning_layers', args_dict,3)
        initialize_parameter(self,'pruning_method', args_dict,2)
        initialize_parameter(self,'pruning_feed_percent', args_dict,0.1)
        initialize_parameter(self,'pruning_flops_threshold', args_dict,0.0001)
        initialize_parameter(self,'pruning_flops_percentage', args_dict,0.5)
        initialize_parameter(self,'pruning_silent', args_dict,True)
        initialize_parameter(self,'pruning_pickle_from', args_dict,'')
        initialize_parameter(self,'pruning_normalize_by_layer', args_dict,False)
        initialize_parameter(self,'pruning_normalize_type', args_dict,2)
        initialize_parameter(self,'seed', args_dict,0)
        initialize_parameter(self,'output_dir', args_dict,'./log')
        initialize_parameter(self,'pruning_protect', args_dict,False)
        initialize_parameter(self,'pruning_broadcast', args_dict,False)
        initialize_parameter(self,'need_hessian', args_dict, False)
        initialize_parameter(self,'hessian_embed', args_dict, 6.0)

        if not self.pruning_iterative:
            self.pruning_frequency = max(0, int(self.pruning_feed_percent * args_dict['max_iter'])-1)
            self.pruning_per_iteration = 0
            self.maximum_pruning_iterations=1
        
class BaseUnitPruner(object):
    """
    pruning base class
    contains pruning config, flops counter of model, importance calculator, and so on.
    note that you should rewrite FLOPs function if you change model, as 
    the FLOPs function is written for deit.
    """
    def __init__(self, model, pruning_parameters_list, pruning_settings, logger, scorer=taylor1Scorer):
        self.masked_model = model
        self.pruning_modules = pruning_parameters_list
        self.pruning_parameters = [m.weight for m in self.pruning_modules]
        self.cfg = copy.deepcopy(pruning_settings)
        self.hessian = [[] for i in range(len(self.pruning_parameters))]
        self.scorer_choose()

        self.res_pruning = 0
        self.iter_step = min(-1,-self.cfg.start_pruning_after_n_iterations)
        self.pruning_iterations_done = 0
        self.iterations_done = 0
        self.pruned_neurons = self.cfg.starting_neuron

        self.all_neuron_units, _ = self._count_number_of_neurons()
        
        self.pruning_scores = {'score':list(),'averaged':list(),'denom':list(),'store_cpu':list()}
        self.prune_network_criteria = list()
        self.pruning_gates = list()
        for layer in range(len(self.pruning_parameters)):
            self.pruning_gates.append(np.ones(len(self.pruning_parameters[layer]),))
            for key in self.pruning_scores.keys():
                self.pruning_scores[key].append(list())
            self.prune_network_criteria.append(np.zeros(len(self.pruning_parameters[layer]),))

        self.get_flops_config()

    def get_flops_config(self):
        self.model_flops_config = {}
        '''
        TODO add automatic flops counter for all models in the future
        get the flops counter for deit
        '''
        self.model_flops_config['layers'] = len(self.masked_model.blocks)
        num_heads = self.masked_model.num_heads * self.model_flops_config['layers']
        num_hidden_neurons = 4 * self.masked_model.embed_dim * self.model_flops_config['layers']
        num_res_neurons = self.masked_model.embed_dim
        self.model_flops_config['nums_components'] = [num_heads, num_hidden_neurons, num_res_neurons]
        self.model_flops_config['length'] = int(197)
        self.model_flops_config['dim'] = self.masked_model.embed_dim 

    def FLOPs(self, x, layer=1):
        '''
        FLOPs counter function for your model.
        '''
        length = self.model_flops_config['length']
        dim = self.model_flops_config['dim']
        layers = self.model_flops_config['layers']
        remain_a = 1 - x[0]
        remain_b = 1 - x[1]
        remain_g = 1 - x[2]
        flops = (length*remain_a + 2*remain_a*remain_g*dim+4*remain_b*remain_g*dim)*2*length*dim*layers
        return flops/1e9

    def scorer_choose(self):
        '''
        used for pruning each comonent individually, such as fisher information metric
        '''
        if self.cfg.pruning_method == 1:
            self.scorer = taylor1Scorer
        elif self.cfg.pruning_method == 2:
            self.scorer = taylor2Scorer
        elif self.cfg.pruning_method == 3:
            self.scorer = taylor3Scorer
        elif self.cfg.pruning_method == 4:
            self.scorer = taylor4Scorer      
        else:
            raise NotImplementedError("this method has not implemented")
    def broadcast_params(self):
        for i in range(len(self.cfg.pruning_stages)):
            for j in range(len(self.pruning_modules[i])):
                self.pruning_modules[i][j].weight.data = self.pruning_parameters[i].data

    def reset(self):
        for layer in range(len(self.pruning_parameters)):
            self.pruning_parameters[layer].data = torch.ones_like(self.pruning_parameters[layer])
            self.pruning_gates[layer] = np.ones(len(self.pruning_parameters[layer]),)
        if self.cfg.pruning_broadcast:
            self.broadcast_params()

    def do_step(self, logger, loss=None):
        self.iter_step += 1
        # stop if pruned maximum amount
        if self.cfg.maximum_pruning_iterations <= self.pruning_iterations_done:
            # exit if we pruned enough
            self.res_pruning = -1
            return -1
        # get scores
        for layer, module in enumerate(self.pruning_modules):
            scores, denoms = self.scorer(module)
            if self.iterations_done == 0:
                self.pruning_scores['score'][layer] = scores
                self.pruning_scores['denom'][layer] = denoms
            else:
                self.pruning_scores['score'][layer] += scores
                self.pruning_scores['denom'][layer] += denoms
        
        self.iterations_done += 1
        if self.iter_step % self.cfg.pruning_frequency == 0 and self.iter_step != 0:
            for layer, score in enumerate(self.pruning_scores['score']):
                if self.cfg.pruning_method == 1:
                    contribution = self.pruning_scores['score'][layer] / self.pruning_scores['denom'][layer]
                else:
                    contribution = self.pruning_scores['score'][layer] / self.iterations_done
                if self.pruning_iterations_done ==0 or not self.cfg.pruning_momentum:
                    self.pruning_scores["averaged"][layer] = contribution
                else:
                    self.pruning_scores["averaged"][layer]=self.cfg.pruning_momentum*self.pruning_scores["averaged"][layer] + \
                        (1-self.cfg.pruning_momentum)*contribution

                current_layer = self.pruning_scores['averaged'][layer].cpu().numpy()
                if self.cfg.pruning_normalize_by_layer:
                    eps = 1e-8
                    if self.cfg.pruning_normalize_type == 1:
                        current_layer = current_layer / (np.linalg.norm(current_layer, ord=1) + eps)
                    elif self.cfg.pruning_normalize_type == 2:
                        current_layer = current_layer / (np.linalg.norm(current_layer) + eps)
                for unit in range(len(self.pruning_parameters[layer])):
                    criterion_now = current_layer[unit].item()
                    # make sure that pruned neurons have 0 criteria
                    current_layer[unit] = criterion_now * self.pruning_gates[layer][unit]
                self.pruning_scores['store_cpu'][layer] = current_layer
            # count number of neurons
            self.criteria = self.pruning_scores['store_cpu']
            if not self.cfg.pruning_silent:
                store_criteria = self.criteria
                pickle.dump(store_criteria, open(self.cfg.output_dir + '/' +
                    "criteria_m{}l{}_final.pickle".format(self.cfg.pruning_method,self.cfg.pruning_layers), "wb"))
            self.iterations_done = 0
            self.pruning_iterations_done += 1

    def enforce_pruning(self):
        for layer, _ in enumerate(self.pruning_parameters):
            for unit in range(len(self.pruning_parameters[layer])):
                if self.pruning_gates[layer][unit] == 0.0:
                    self.pruning_parameters[layer].data[unit] *= 0.0   
        if self.cfg.pruning_broadcast:
            self.broadcast_params()
    
    def get_criteria(self):
        if hasattr(self, 'criteria'):
            return self.criteria
        if len(self.cfg.pruning_pickle_from) > 0:
            pruning_mask_from = self.cfg.pruning_pickle_from + '/' + \
                "criteria_m{}l{}_final.pickle".format(self.cfg.pruning_method,self.cfg.pruning_layers)
        else:
            pruning_mask_from = self.cfg.output_dir + '/' + \
                "criteria_m{}l{}_final.pickle".format(self.cfg.pruning_method,self.cfg.pruning_layers)
        criteria = pickle.load(open(pruning_mask_from, "rb"))
        self.criteria = criteria
        return criteria

    def prune(self, logger, criteria=None, threshold_now=None,show=True):
        self.prune_network_criteria = criteria
        # adaptively estimate threshold given a number of neurons to be removed
        if threshold_now is None:
            all_criteria = np.asarray([criteria for layer in self.prune_network_criteria for criteria in layer]).reshape(-1)
            prune_neurons_now = min(len(all_criteria)-1, self.cfg.pruning_per_iteration - 1)
            threshold_now = np.sort(all_criteria)[prune_neurons_now]
        if type(threshold_now)!=list:
            threshold_now=[threshold_now]*len(self.prune_network_criteria)
        for layer in range(len(self.prune_network_criteria)):
            if show:
                logger.info("\nLayer:{}".format(layer))
                logger.info("units:{}".format(len(self.prune_network_criteria[layer])))

            index = np.where(self.prune_network_criteria[layer]<=threshold_now[layer])
            self.pruning_gates[layer][index] *= 0.0
            self.pruning_parameters[layer].data[index] *= 0.0
            # for unit, criteria in enumerate(self.prune_network_criteria[layer]):
            #     if criteria <= threshold_now:
            #         # do actual pruning
            #         self.pruning_gates[layer][unit] *= 0.0
            #         self.pruning_parameters[layer].data[unit] *= 0.0
            if show:
                logger.info("pruned_perc:{}".format([np.nonzero(1.0-self.pruning_gates[layer])[0].size, len(self.pruning_parameters[layer])]))   
        if self.cfg.pruning_broadcast:
            self.broadcast_params()
        # count number of neurons
        if show:
            all_neuron_units, neuron_units = self._count_number_of_neurons()
            self.pruned_neurons = all_neuron_units-neuron_units
            logger.info("---------------pruning information---------")
            logger.info("method:{}".format(self.cfg.pruning_method))
            logger.info("pruning layer:{}".format(self.cfg.pruning_layers))
            logger.info("pruned_neurons:{}".format(self.pruned_neurons))
            logger.info("pruning_iterations_done:{}".format(self.pruning_iterations_done))
            logger.info("neuron_units:{}".format(neuron_units))
            logger.info("all_neuron_units:{}".format(all_neuron_units))
            logger.info("threshold_now:{}".format(threshold_now))

        self.threshold_now = threshold_now   
        # set result to successful
        self.res_pruning = 1
    
    def _count_number_of_neurons(self):
        all_neuron_units = 0
        neuron_units = 0
        for layer, _ in enumerate(self.pruning_parameters):

            all_neuron_units += len( self.pruning_parameters[layer])
            for unit in range(len(self.pruning_parameters[layer])):
                if len(self.pruning_parameters[layer].data.size()) > 1:
                    statistics = self.pruning_parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.pruning_parameters[layer].data[unit]

                if statistics > 0.0:
                    neuron_units += 1
        return all_neuron_units, neuron_units
        
    def protect(self, model_without_ddp, logger, every_per=0.05, percentage=None):
        '''
        ensure every layer must contains some percentage of neurons to prevent
        performance collapse.
        Args:
            every_per: every layer percentage for protect
            percentage: the whold percentage for protect
        '''
        criteria = self.get_criteria()
        # assure all the layers connected with all shortcut retain neurons above minimum percentage 
        import warnings
        warnings.warn('we will keep minimum {}% neurons in every layers'.format(every_per*100))
        # --------assure percentage of every layer
        for row in range(len(criteria)):
            # reset the every_per neurons importance to big value
            keep_size = max(int(len(criteria[row])*every_per),1)
            ind = np.argpartition(criteria[row],-keep_size)[-keep_size:]
            ind = np.argpartition(criteria[row],-keep_size)[0:-keep_size]
            criteria[row] = criteria[row][ind]
        if not percentage:
            self.criteria=criteria
            return criteria
        
        # TODO protect by percentage
        criteria = self.get_criteria()
        warnings.warn('we will keep minimum {}% neurons in total'.format(percentage*100))
        return criteria
        
    def importance_show(self, criteria):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        i_matrix_list = criteria
        max_length = max(i_matrix.size for i_matrix in i_matrix_list)
        for i in range(len(i_matrix_list)):
            i_matrix_list[i] = np.pad(i_matrix_list[i], 
                                 (0, max_length-i_matrix_list[i].size),
                                 constant_values=(0, 0))
        i_matrix = np.stack(i_matrix_list)
        ax.matshow(i_matrix, cmap=plt.cm.Reds)
        file_name = self.output_dir + '/' + 'im_pruning_layer{}_method{}.png'.format(self.pruning_layers,self.method)
        fig.suptitle('im_pruning_layer{}_method{}.png'.format(self.pruning_layers,self.method))
        plt.show()      
        plt.savefig(file_name)

'''
evolution algorithm default setting
'''
DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 1]       # solution upper and lower bounds
N_GENERATIONS = 100      # search iterations
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation

class Controller(object):
    """
    pruning controller class
    conduct evolution algorithm to search pruning target of all components, 
    then conduct pruning in each component.
    """
    def __init__(self, pruning_engines):
        self.pruning_engines = pruning_engines
        if pruning_engines is not None:
            self.need_hessian = pruning_engines[0].cfg.need_hessian
            self.cfg = self.pruning_engines[0].cfg
            self.num_layers = len(pruning_engines[0].pruning_gates)
            self.num_com, num_com = len(self.pruning_engines), len(self.pruning_engines)
            self.second_coef = torch.zeros((num_com, num_com)).to(next(self.pruning_engines[0].masked_model.parameters()).device)
            embed_dim = self.pruning_engines[0].masked_model.embed_dim
            num_heads = self.pruning_engines[0].masked_model.num_heads
            hessian_embed = self.cfg.hessian_embed
            a = embed_dim    # as each head contains feature with embed_dim, so embed_dim x num_heads
            b = embed_dim*4   
            c = embed_dim*self.num_layers*hessian_embed # as each embedding dim is connected to all sub-blocks
            self.num_coms = torch.tensor([
                [a*a,                a*b,             a*c],
                [b*a,                b*b,             b*c],
                [c*a,                c*b,             c*c]
            ]).to(next(self.pruning_engines[0].masked_model.parameters()).device)
            self.count = 0
            self.re_compute = True
            self.load_hessian()

    def compute_hessian(self):
        if not self.re_compute:  return
        first_layer = torch.randint(0, self.num_layers, (1,))[0]
        for i in range(self.num_com):
            if i == 2:
                first_layer = 0
            first_order = self.pruning_engines[i].pruning_modules[first_layer].weight.grad
            vec = self.pruning_engines[i].pruning_modules[first_layer].weight
            for j in range(self.num_com):
                if j == 2:
                    second_layer = 0
                else:
                    second_layer = max(0, torch.randint(0, first_layer+1, (1,))[0])
                weight = self.pruning_engines[j].pruning_modules[second_layer].weight
                second_order = torch.autograd.grad(first_order, weight, vec, retain_graph=True)[0]
                self.second_coef[i,j] += second_order.sum()/self.num_coms[i,j]
        self.count += 1
    
    def load_hessian(self):
        # save or load the hessian-vector
        if len(self.cfg.pruning_pickle_from) > 0:
            pruning_hessian_from = self.cfg.pruning_pickle_from + "/hessian.pickle"
            self.re_compute = False
            if os.path.exists(pruning_hessian_from):
                self.second_coef = pickle.load(open(pruning_hessian_from, "rb"))

    def save_hessian(self):
        if self.re_compute:
            self.second_coef = self.second_coef / (self.count + 1e-9)
            self.second_coef = self.second_coef.cpu().numpy()
            pickle.dump(self.second_coef, open(self.cfg.output_dir + "/hessian.pickle", "wb"))
            self.re_compute = False

    def _normalize_scores(self, criteria):
        """
        LAMP normalizing scheme for importance metric.
        for details please seee <Layer-adaptive sparsity for the Magnitude-based Pruning>
        """
        import torch
        for i, scores in enumerate(criteria):
            # sort scores in an ascending order
            scores = torch.tensor(scores)
            sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
            # compute cumulative sum
            scores_cumsum_temp = sorted_scores.cumsum(dim=0)
            scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
            scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
            # normalize by cumulative sum
            sorted_scores /= (scores.sum() - scores_cumsum)
            scores[sorted_idx]=sorted_scores
            criteria[i] = scores.numpy()
            
        return criteria

    def _l1_normalize_per_layer(self, criteria):
        """
        L1 normalizing scheme for importance metric.
        """
        eps = 1e-9
        for i, scores in enumerate(criteria):
            denom = np.linalg.norm(scores, ord=1)
            criteria[i] = (criteria[i])/(denom+eps)
        return criteria

    def _l2_normalize_per_layer(self, criteria):
        """
        L2 normalizing scheme for importance metric.
        """
        eps = 1e-9
        for i, scores in enumerate(criteria):
            denom = np.linalg.norm(scores)
            criteria[i] = (criteria[i])/(denom+eps)
        return criteria

    def _gauss_normalize(self, criteria):
        """
        Gaussian normalizing scheme for importance metric.
        """
        criteria_flatten = np.asarray([c for layer in criteria for c in layer]).reshape(-1)
        mu = np.mean(criteria_flatten)
        sigma = np.var(criteria_flatten)
        eps = 1e-9
        for i in range(len(criteria)):
            criteria[i] = (criteria[i]-mu)/(sigma+eps)
        return criteria

    def _square(self, criteria):
        """
        square normalizing scheme for importance metric.
        """
        for i in range(len(criteria)):
            criteria[i] = np.square(criteria[i])
        return criteria

    def threshold_find(self, criterias, num_pruned, max_ratio=False, con_sums=None):
        '''
        depricated function
        '''
        sorted_criterias = []
        sum_criterias = []
        if con_sums is None:
            con_sums = []
        for criteria in criterias:
            flatten_c = np.asarray([c for layer in criteria for c in layer]).reshape(-1)
            sorted_c = np.sort(flatten_c)
            sorted_criterias.append(sorted_c)
            sum_c = np.cumsum(sorted_c)
            sum_criterias.append(sum_c)
            if con_sums is None:
                con_sums.append(max(sum_c))

        thresholds = [[]] * len(criterias)
        indexes = [[]] * len(criterias)
        thresholds[1]=sorted_criterias[1][num_pruned]
        indexes[1]=num_pruned
        cum_fc=sum_criterias[1][num_pruned]
        con_sum_fc = con_sums[1]
        for i, (sorted_c, sum_c, con_sum) in enumerate(zip(sorted_criterias,sum_criterias, con_sums)):
            if i== 1:
                continue
            if max_ratio:
                cur_sum = cum_fc/con_sum_fc * con_sum
            else:
                cur_sum = cum_fc
            index = np.argmax(sum_c>cur_sum)
            index=max(0,index-1)
            indexes[i] = index
            thresholds[i] = sorted_c[index]
        criterias_str=['head','fc','res','patch', '.', '.']
        s='pruning: '
        for i in range(len(criterias)):
            s += criterias_str[i] +':{} '.format(indexes[i])
        print(s)
        return thresholds

    def pruning_x(self, model_without_ddp, logger, max_ratio=False,
                pruning_flops_percentage=None, pruning_flops_threshold=None):
        '''
        depricated method
        '''
        protect=self.pruning_engines[0].cfg.pruning_protect
        # find a way to prune head and ffn more effectively
        assert len(self.pruning_engines)>1
        if pruning_flops_percentage is None:
            pruning_flops_percentage = self.pruning_engines[0].cfg.pruning_flops_percentage
        if pruning_flops_threshold is None:
            pruning_flops_threshold = self.pruning_engines[0].cfg.pruning_flops_threshold
        
        con_sums = []
        criterias = []
        for pruning_engine in self.pruning_engines:
            if protect:
                pruning_engine.protect(model_without_ddp, logger)
                logger.info('------------protect layer {} done----------------'.format(pruning_engine.cfg.pruning_layers))
            else:
                logger.info('------------no protect for layer {}----------------'.format(pruning_engine.cfg.pruning_layers))
            current = pruning_engine.get_criteria()
            criterias.append(current)
            con_sums.append(sum([c.sum() for c in current]))
            if pruning_engine.cfg.pruning_normalize_by_layer:
                current=self._l2_normalize_per_layer(current)

        current_flops = list()
        start = 0
        end = self.pruning_engines[1].all_neuron_units-1
        mid = (start + end + 1 ) // 2
        from functools import partial
        _threshold_find=partial(self.threshold_find,max_ratio=max_ratio, con_sums=con_sums)
        while 1:
            thresholds=_threshold_find(criterias, num_pruned=mid)
            for i, pruning_engine in enumerate(self.pruning_engines):
                pruning_engine.reset()
                pruning_engine.prune(logger, pruning_engine.criteria,threshold_now=thresholds[i])

            current_flops = get_model_complexity_info(model_without_ddp, logger)
            pruned_flops =  self.pruning_engines[0].cfg.flops[-1] - current_flops[-1]
            pruned_percentage = pruned_flops/self.pruning_engines[0].cfg.flops[-1]
            if start>=end:
                break
            if pruned_percentage < (pruning_flops_percentage - pruning_flops_threshold) and (start != mid):
                start = mid
                mid = (start + end) // 2
            elif pruned_percentage > (pruning_flops_percentage + pruning_flops_threshold) and (end != mid):
                end = mid
                mid = (start + end) // 2
            else:
                break

    def F(self, x, h_cumsum, fc_cumsum, res_cumsum, nums_components): 
        '''
        the object of ea algorithm is to find the largest target.
        '''
        a = x[0]/nums_components[0]
        b = x[1]/nums_components[1]
        c = x[2]/nums_components[2]
        ratio = self.second_coef[0,0]*(a)**2 + self.second_coef[1,1]*(b)**2 + self.second_coef[2,2]*(c)**2+ \
                self.second_coef[0,1]*a*b*2 + self.second_coef[0,2]*a*c*2 + self.second_coef[1,2]*b*c*2
        temp = -ratio
        return temp

    def get_fitness(self, pred): 
        '''
        find non-zero fitness for selection
        '''
        return pred

    def make_kid(self, pop, n_kid):
        '''
        generate empty kid holder
        '''
        kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for kv, ks in zip(kids['DNA'], kids['mut_strength']):
            # crossover (roughly half p1 and half p2)
            p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
            cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
            kv[cp] = pop['DNA'][p1, cp]
            kv[~cp] = pop['DNA'][p2, ~cp]
            ks[cp] = pop['mut_strength'][p1, cp]
            ks[~cp] = pop['mut_strength'][p2, ~cp]

            # mutate (change DNA based on normal distribution)
            ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
            kv += ks * np.random.randn(*kv.shape)
            kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
        return kids

    def kill_bad(self, pop, kids, fitness):
        '''
        kill inappropriate childs.
        '''
        idx = np.arange(pop['DNA'].shape[0])
        good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
        for key in ['DNA', 'mut_strength']:
            pop[key] = pop[key][good_idx]
        return pop

    def pruning_ea(self, model_without_ddp, logger,
                pruning_flops_percentage=None, pruning_flops_threshold=None):
        '''
        using ea algorithm to search pruning ratios.
        '''
        protect=self.pruning_engines[0].cfg.pruning_protect
        # save evolution algorithm setting
        logger.info('DNA_SIZE:{}'.format(DNA_SIZE))
        logger.info('POP_SIZE:{}'.format(POP_SIZE))
        logger.info('N_KID:{}'.format(N_KID))
        logger.info('N_GENERATIONS:{}'.format(N_GENERATIONS))
        logger.info('DNA_BOUND:{}'.format(DNA_BOUND))       
        # find a way to prune head and ffn more effectively
        assert len(self.pruning_engines)>1
        if pruning_flops_percentage is None:
            pruning_flops_percentage = self.pruning_engines[0].cfg.pruning_flops_percentage
        if pruning_flops_threshold is None:
            pruning_flops_threshold = self.pruning_engines[0].cfg.pruning_flops_threshold
        for pruning_engine in self.pruning_engines:
            current = pruning_engine.get_criteria()
            if pruning_engine.cfg.pruning_normalize_by_layer:
                current=self._l2_normalize_per_layer(current)
            if protect:
                pruning_engine.protect(model_without_ddp, logger)
                logger.info('------------protect layer {} done----------------'.format(pruning_engine.cfg.pruning_layers))
            else:
                logger.info('------------no protect for layer {}----------------'.format(pruning_engine.cfg.pruning_layers))

        head_criteria=self.pruning_engines[0].get_criteria()
        fc_criteria=self.pruning_engines[1].get_criteria()
        res_criteria=self.pruning_engines[2].get_criteria()
        head_flatten_criteria = np.asarray([c for layer in head_criteria for c in layer]).reshape(-1)
        fc_flatten_criteria = np.asarray([c for layer in fc_criteria for c in layer]).reshape(-1)
        res_flatten_criteria= np.asarray([c for layer in res_criteria for c in layer]).reshape(-1)
        head_criteria = np.sort(head_flatten_criteria)
        fc_criteria = np.sort(fc_flatten_criteria)
        res_criteria = np.sort(res_flatten_criteria)
        h_cumsum=np.insert(np.cumsum(head_criteria),0,0)
        fc_cumsum=np.insert(np.cumsum(fc_criteria),0,0)
        res_cumsum=np.insert(np.cumsum(res_criteria),0,0)
        head_pop = dict(DNA=np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),  
                        mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))               
        res_pop =  dict(DNA=np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),  
                        mut_strength=np.random.rand(POP_SIZE, DNA_SIZE)) 
        head = np.random.randint(1, size=POP_SIZE)   # initialize the pop DNA
        fc = np.random.randint(1, size=POP_SIZE)   # initialize the pop DNA
        res = np.random.randint(1, size=POP_SIZE)   # initialize the pop DNA
        head = np.random.randint(1, size=POP_SIZE+N_KID)   # initialize the pop DNA
        fc = np.random.randint(1, size=POP_SIZE+N_KID)   # initialize the pop DNA
        res = np.random.randint(1, size=POP_SIZE+N_KID)   # initialize the pop DNA
        index = 0
        head_total = len(head_criteria)-1
        res_total = len(res_criteria)-1
        fc_total = len(fc_criteria)-1
        pruned_per = np.zeros(3)
        total_flops = self.pruning_engines[0].FLOPs(pruned_per)
        nums_components = self.pruning_engines[0].model_flops_config['nums_components']
        for _ in range(N_GENERATIONS):
            head_kids = self.make_kid(head_pop, N_KID)
            res_kids = self.make_kid(res_pop, N_KID)   
            for key in ['DNA', 'mut_strength']:
                head_pop[key] = np.vstack((head_pop[key], head_kids[key]))
                res_pop[key] = np.vstack((res_pop[key], res_kids[key]))
            for i in range(POP_SIZE+N_KID):
                head[i]=int(head_total*head_pop['DNA'][i])
                res[i] = int(res_total*res_pop['DNA'][i])
                head_th=head_criteria[head[i]]
                res_th=res_criteria[res[i]]
                current_flops = list()
                start = 0
                end = fc_total
                mid = (start + end + 1 ) // 2

                pruned_per = [head[i]/nums_components[0], 0, res[i]/nums_components[2]]
                current_flops = self.pruning_engines[0].FLOPs(pruned_per)
                pruned_flops =  total_flops - current_flops
                pruned_percentage = pruned_flops/total_flops
                if pruned_percentage > pruning_flops_percentage :
                    fc[i]=0
                    continue
                while 1:
                    pruned_per = [head[i]/nums_components[0], mid/nums_components[1], res[i]/nums_components[2]]
                    current_flops = self.pruning_engines[0].FLOPs(pruned_per)
                    pruned_flops =  total_flops - current_flops
                    pruned_percentage = pruned_flops/total_flops
                    if start>=end:
                        break
                    if pruned_percentage < (pruning_flops_percentage - pruning_flops_threshold) and (start + 1 <= mid):
                        start = mid
                        mid = (start + end) // 2
                    elif pruned_percentage > (pruning_flops_percentage + pruning_flops_threshold) and (end - 1 >= mid):
                        end = mid
                        mid = (start + end) // 2
                    else:
                        break
                fc[i]=mid
                
            chromosome=[head,fc,res]
            F_values = self.F(chromosome,h_cumsum,fc_cumsum,res_cumsum, nums_components)
            fitness = self.get_fitness(F_values)
            index = np.argmax(fitness)
            print("Most fitted DNA is head: {}, fc:{}. res:{} ".format(head[index],fc[index], res[index]))
            head_pop = self.kill_bad(head_pop, head_kids, fitness)   # keep some good parent for elitism
            res_pop = self.kill_bad(res_pop, res_kids, fitness)   # keep some good parent for elitism
        
        thresholds=[head_criteria[head[index]],fc_criteria[fc[index]],res_criteria[res[index]]]
        for i, pruning_engine in enumerate(self.pruning_engines):
            pruning_engine.reset()
            pruning_engine.prune(logger, pruning_engine.criteria,threshold_now=thresholds[i])