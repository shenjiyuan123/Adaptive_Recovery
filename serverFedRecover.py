"""
Implementation of paper: FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information
"""
import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import scipy
import json
import wandb

from serverBase import Server
from clientBase import clientFedRecover


class FedRecover(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRecover)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number
        
        
    def train(self):
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
                self.evaluate()
                print("\n")
                # self.server_metrics()
            # print(self.selected_clients)
            for client in self.selected_clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    client.train(create_trigger=True)
                elif client in self.unlearn_clients and self.trim_attack:
                    client.train(trim_attack=True)
                else:
                    client.train()
            
            self.save_client_model(i)

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_models()
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=False, trimmed_clients_num=self.args.trimmed_clients_num)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=False)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=False)

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
        # self.server_metrics()
        self.FL_global_model = copy.deepcopy(self.global_model)



    def recover(self):
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.initial_model)
        for client in self.remaining_clients:
            client.set_parameters(self.global_model)
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        prev_train_loss = 10

        for i in range(self.global_rounds+1):
            s_t = time.time()

            assert (len(self.clients) > 0)
            for client in self.remaining_clients:
                start_time = time.time()
                
                client.set_parameters(self.global_model)
                
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
                

            if i%self.eval_gap == 0:
                print(f"\n-------------FedRecover Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc = self.evaluate()
                wandb.log({f'Train_loss/{self.algorithm}': train_loss}, step=i)
                wandb.log({f'Test_acc/{self.algorithm}': test_acc}, step=i)
            
                # self.server_metrics()
            
            if train_loss > prev_train_loss:
                self.load_epoch_GModel(i)
                print('load global again.')
            prev_train_loss = train_loss    
                
            # print(self.remaining_clients, len(self.remaining_clients), len(self.unlearn_clients))
            for client in self.remaining_clients:
                client.retrain_with_LBFGS()

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_retrained_models(self.remaining_clients)
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.remaining_clients)
            
            
            # print("retrain ***:::", self.global_model.state_dict()['base.conv1.0.weight'][0])

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
        self.save_global_model()
        # self.server_metrics()
        self.eraser_global_model = copy.deepcopy(self.global_model)



