import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server



class FedEraser(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number


    def train(self):
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
            

        
    def unlearning(self):
        print("***************", self.unlearn_clients)
        
        model_path = os.path.join("server_models", self.dataset)
        
        for epoch in range(0, self.global_rounds, 1):
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert (os.path.exists(server_path))
            self.old_GM = torch.load(server_path)
            # print("old GM ***:::", self.old_GM.state_dict()['base.conv1.0.weight'][0])
            
            all_clients_class = self.load_client_model(epoch)
            for i, client in enumerate(self.remaining_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
                        # print(" /// ",c.model.state_dict()['base.conv1.0.weight'][0])
            self.old_CM = copy.deepcopy(self.remaining_clients)


            # 产生第一次的new_GM
            if epoch == 0:
                weight = []
                for c in self.remaining_clients:
                    weight.append(c.train_samples)
                tot_sample = sum(weight)
                weight = [i / tot_sample for i in weight]
                pprint(weight)
            
                for param in self.global_model.parameters():
                    param.data.zero_()
                for w, client in zip(weight, self.remaining_clients):
                    self.add_parameters(w, client.model)
                self.new_GM = copy.deepcopy(self.global_model)
                # print(self.new_GM.state_dict()['base.conv1.0.weight'][0])
                
                continue
            
            print(f"\n-------------FedEraser Round number: {epoch}-------------")
            train_loss, test_acc = self.evaluate()
            wandb.log({f'Train_loss/{self.algorithm}': train_loss}, step=epoch)
            wandb.log({f'Test_acc/{self.algorithm}': test_acc}, step=epoch)
                
            # 得到新的CM，进行一步训练
            assert (len(self.remaining_clients) > 0)
            for client in self.remaining_clients:
                client.set_parameters(self.new_GM)
                client.train()
            self.new_CM = copy.deepcopy(self.remaining_clients)
            
            # 聚合一次
            self.receive_retrained_models(self.remaining_clients)
            self.aggregate_parameters()
            self.new_GM = copy.deepcopy(self.global_model)
            # print("New_GM before calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
            
            # 开始校准
            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
            # print("new GM after calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
        
        print(f"\n-------------After FedEraser-------------")
        print("\nEvaluate Eraser globel model")
        self.server_metrics()
        self.eraser_global_model = copy.deepcopy(self.new_GM)
            
    
    def unlearning_step_once(self, old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
        """
        Parameters
        ----------
        old_client_models : list of client objects
        
        new_client_models : list of client objects
            
        global_model_before_forget : The old global model
            
        global_model_after_forget : The New global model
            

        Returns
        -------
        return_global_model : After one iteration, the new global model under the forgetting setting

        """
        old_param_update = dict()#Model Params： oldCM - oldGM_t
        new_param_update = dict()#Model Params： newCM - newGM_t
        
        new_global_model_state = global_model_after_forget.state_dict()#newGM_t
        
        return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
        
        assert len(old_client_models) == len(new_client_models)
        
        for layer in global_model_before_forget.state_dict().keys():
            old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            for ii in range(len(new_client_models)):
                old_param_update[layer] += old_client_models[ii].model.state_dict()[layer]
                new_param_update[layer] += new_client_models[ii].model.state_dict()[layer]
            old_param_update[layer] /= (ii+1)#Model Params： oldCM
            new_param_update[layer] /= (ii+1)#Model Params： newCM
            
            old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]#参数： oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]#参数： newCM - newGM_t
            
            step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
            step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
            
            return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction
        
        # print("....", step_length, step_direction)
        return_global_model = copy.deepcopy(global_model_after_forget)
        
        return_global_model.load_state_dict(return_model_state)
        
        return return_global_model


