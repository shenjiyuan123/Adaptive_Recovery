import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import json
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from serverEraser import FedEraser


class Crab(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 'client_selection': []
        self.info_storage = {}
        self.new_CM = []
        self.P_rounds = args.select_round_ratio
        self.X_clients = args.select_client_ratio
    

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        if self.new_CM != []:
            for c in self.new_CM:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
            ids = [c.id for c in self.new_CM]
        else:
            for c in self.remaining_clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
            ids = [c.id for c in self.remaining_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        asrs = []
        if self.new_CM != []:
            for c in self.new_CM:
                casr, cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl*1.0)
                asrs.append(casr*1.0)

            ids = [c.id for c in self.new_CM]
        else:
            for c in self.remaining_clients:
                casr, cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl*1.0)
                asrs.append(casr*1.0)

            ids = [c.id for c in self.remaining_clients]

        return ids, num_samples, losses
    
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        if self.remaining_clients:
            stats_target = self.target_metrics()
            train_asr = sum(stats_target[3])*1.0 / sum(stats_target[1])

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
        if self.remaining_clients:
            print("Averaged Attack success rate: {:.4f}".format(train_asr))
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
            # print(sum(p.numel() for p in model.parameters()))
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
        k = int(len(GM_list) * self.P_rounds) 
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
        k = int(len(CM) * self.X_clients)
        target_GM = GM_list[round - start_epoch] # GM_list 的下标是根据每一个 buffer window 从0开始索引的
        target_GM = [p.detach().clone() for p in target_GM.parameters()]

        similarity = []
        for client in CM_list:  
            cos_sim = [] 
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    # TODO: 是否考虑 abs(cos)
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
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        alpha = 0.1
        GM_list = []
        start_epoch = 0
        
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)
            

            if i%self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, _ = self.evaluate()
                
            if i == 0:
                start_loss = copy.deepcopy(train_loss) 
            else:
                GM_list.append(copy.deepcopy(self.global_model))
                
            if train_loss < start_loss * (1 - alpha) or i == self.global_rounds:
            # if i%5==0:
                print("*****")
                rounds = self.select_round(start_epoch, GM_list)
                print("pick rounds: ", rounds)
                for round in rounds:
                    clients_id = self.select_client_in_round(round, GM_list, start_epoch)
                    print(f"select clients from epoch {round}: {clients_id}")
                    self.info_storage[int(round)] = clients_id
                    
                # for client in clients_id:
                    # gradient = self.select_grad_in_client()
                
                start_loss = copy.deepcopy(train_loss)
                GM_list = []
                start_epoch = i
                # self.info_storage = ...
                
            for client in self.selected_clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    client.train(create_trigger=True)
                elif client in self.unlearn_clients and self.trim_attack:
                    client.train(trim_attack=True)
                else:
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


        print('write the select information into the txt...')
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        path = os.path.join(self.save_folder_name, "server_select_info" + ".txt")
        self.info_storage = dict(sorted(self.info_storage.items()))
        with open(path, 'w') as storage: 
            storage.write(json.dumps(self.info_storage))
            
        self.save_results()
        # self.save_global_model()
        self.server_metrics()
        self.FL_global_model = copy.deepcopy(self.global_model)


    def adaptive_recover(self):
        print("***************", self.unlearn_clients)
        
        model_path = os.path.join("server_models", self.dataset)
        
        for global_round, select_clients_in_round in self.info_storage.items():
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(global_round) + ".pt")
            self.old_GM = torch.load(server_path)
            
            select_clients_in_round = [id for id in select_clients_in_round if id in self.idr_]
            
            all_clients_class = self.load_client_model(global_round)
            # 此处copy一份remaining_clients，因为recovery的时候可能遗忘的id包含贡献度大的那个client
            self.old_clients = copy.deepcopy(self.remaining_clients)  
            self.old_CM = []
            for i, client in enumerate(self.old_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
                        # print(" /// ",c.model.state_dict()['base.conv1.0.weight'][0])
                if client.id in select_clients_in_round:
                    self.old_CM.append(client)
            print([c.id for c in self.old_CM])
            
            self.old_clients = copy.deepcopy(self.old_CM)
        
            
            # 得到新的GM
            assert (len(self.old_CM) <= len(select_clients_in_round))
            for client in self.old_clients:
                client.set_parameters(self.old_GM)
                client.train_one_step()

            self.receive_retrained_models(self.old_clients)
            self.aggregate_parameters()
            self.new_GM = copy.deepcopy(self.global_model)
            # print("New_GM before calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
            
            # 得到新的CM
            for client in self.old_clients:
                client.set_parameters(self.new_GM)
                client.train_one_step()
            self.new_CM = copy.deepcopy(self.old_clients)
            
            print(f"\n-------------Crab Round number: {global_round}-------------")
            
            # test_acc, test_loss = self.server_metrics()
            # wandb.log({f'Train_loss/{self.algorithm}': test_loss}, step=global_round)
            train_loss, test_acc = self.evaluate()
            wandb.log({f'Train_loss/{self.algorithm}': train_loss}, step=global_round)
            wandb.log({f'Test_acc/{self.algorithm}': test_acc}, step=global_round)
            
            # 开始校准
            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
            # print("new GM after calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
        
        print(f"\n-------------After Crab-------------")
        print("\nEvaluate Eraser globel model")
        self.server_metrics()
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.new_CM = []

