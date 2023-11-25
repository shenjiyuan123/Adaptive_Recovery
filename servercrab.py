import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
from pprint import pprint

from dataset_utils import read_client_data
from serverbase import FedAvg, Server
from clientbase import clientAVG


class Crab(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.info_storage = {
            'round_selection': [],
            'client_selection': {},
            'grad_selection': {}
        }
    
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        
        return train_loss, test_acc
    
    
    def model_to_traj(self, GM_list):
        """输入 GM list 返回去除 dict keys 的纯 tensor

        Args:
            GM_list (list of NN)

        Returns:
            list of tensor: GM 随着epoch的trajectory
        """
        traj = []
        for model in GM_list:
            timestamp = []
            timestamp.extend([p.detach().clone() for p in model.parameters()])
            traj.append(timestamp)
        return traj
    
    def select_round(self, start_epoch, GM_list):
        """选取哪些 epoch 最重要

        Args:
            start_epoch (int): 每个 buffer window 的起点
            GM_list (list of NN): [self.global_model]

        Returns:
            list: 返回需要选取的 epoch 轮数
        """
        k = int(len(GM_list) * 0.5) 
        GM_trajectory = self.model_to_traj(GM_list)
        prior = GM_trajectory[0]
        kl_list = []
        if len(GM_trajectory) < 2:
            return [start_epoch]
        for now_traj in GM_trajectory[1:]:
            kl = 0
            for module, prior_module in zip(now_traj, prior):
                log_x = F.log_softmax(module, dim=-1)
                y = F.softmax(prior_module, dim=-1)
                kl += F.kl_div(log_x, y, reduction='sum')
            kl_list.append(kl.cpu().item())
            prior = now_traj
        print("KL Divergence between each epoch's global model:", kl_list)
        kl_list = np.array(kl_list)
        sel_round = np.argsort(kl_list)[::-1]
        return (sel_round[:k] + start_epoch).tolist()
        
        
    def select_client_in_round(self, round, GM_list, start_epoch):
        """在每一个 epoch 中选取需要的 clients

        Args:
            round (int): 当前epoch
            GM_list (list of NN): _description_
            start_epoch (int): buffer window 起始epoch

        Returns:
            list: 返回需要选取的 client id
        """
        CM = self.load_client_model(round)
        CM_list = [c.model for c in CM]
        CM_list = self.model_to_traj(CM_list)
        k = int(len(CM) * 0.5)
        target_GM = GM_list[round - start_epoch] # GM_list 的下标是根据每一个 buffer window 从0开始索引的
        target_GM = [p.detach().clone() for p in target_GM.parameters()]

        similarity = []
        for client in CM_list:  
            cos_sim = [] 
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    cos_sim.append(torch.mean(cos).cpu().item())
            similarity.append(np.mean(cos_sim))
        sel_client = np.argsort(similarity)[::-1]
        sel_client = sel_client[:k].tolist()
        
        ans_clients = []
        ans_id = []
        for sel in sel_client:
            ans_id.append(CM[sel].id)
            # ans_clients.append(CM[sel])

        return ans_id


    def train_with_select(self):
        print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        alpha = 0.1
        GM_list = []
        start_epoch = 0
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)
            

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, _ = self.evaluate()
                
            if i == 0:
                start_loss = copy.deepcopy(train_loss) 
            else:
                GM_list.append(copy.deepcopy(self.global_model))
                
            if train_loss < start_loss * (1 - alpha):
            # if i%5==0:
                print("*****")
                rounds = self.select_round(start_epoch, GM_list)
                print("pick rounds: ", rounds)
                for round in rounds:
                    clients_id = self.select_client_in_round(round, GM_list, start_epoch)
                    print("select clients from epoch {round}: ", clients_id)
                # for client in clients_id:
                    # gradient = self.select_grad_in_client()
                
                start_loss = copy.deepcopy(train_loss)
                GM_list = []
                start_epoch = i
                # self.info_storage = ...
                

            for client in self.selected_clients:
                client.train()
            
            self.save_client_model(i)

            self.receive_models()
            self.aggregate_parameters()
            

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()
        self.server_metrics()
        self.FL_global_model = copy.deepcopy(self.global_model)


