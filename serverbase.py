import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from dataset_utils import read_client_data
from clientbase import clientAVG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.current_round = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def send_specific_models(self, global_model):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
            

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
        
    def save_each_round_global_model(self, epoch):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server_" + str(epoch) + ".pt")
        torch.save(self.global_model, model_path)
        
    def save_client_model(self, epoch):
        model_path = os.path.join("client_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
        torch.save(self.uploaded_models, model_path)

    def load_model(self, epoch):
        if epoch == self.global_rounds:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
            assert (os.path.exists(model_path))
            self.global_model = torch.load(model_path)
        else:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server_" + str(epoch) + ".pt")
            assert (os.path.exists(model_path))
            self.global_model = torch.load(model_path)
            
    
    def load_client_model(self, epoch):
        model_path = os.path.join("client_models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
        assert (os.path.exists(model_path))
        client_models = torch.load(model_path)
        assert (len(client_models) == self.num_clients)
        return client_models

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
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
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
################################################################################################################   
    
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = 1


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.save_client_model(i)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.save_each_round_global_model(i)

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
        self.FL_global_model = copy.deepcopy(self.global_model)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
            
    def unlearning(self):
        idx = [i for i in range(self.num_clients)]
        idx_ = random.sample(range(len(idx)), self.unlearn_clients_number)
        idr_ = [i for i in range(self.num_clients) if i not in idx_]
        self.unlearn_clients = [self.selected_clients[i] for i in idx_]
        self.remaining_clients = [self.selected_clients[i] for i in idr_]
        
        
        for epoch in range(self.global_rounds + 1):
            self.load_model(epoch)
            self.old_GM = copy.deepcopy(self.global_model)
            
            all_clients_model = self.load_client_model(epoch)
            for tmp in idx_:
                all_clients_model.pop(tmp)
            remaining_clients_model = all_clients_model
            
            # 产生第一次的new_GM
            if epoch == 0:
                for i, client in enumerate(self.remaining_clients):
                    client.set_parameters(remaining_clients_model[i])
                self.new_GM = copy.deepcopy(self.old_GM)
                for param in self.new_GM.parameters():
                    param.data.zero_()
                weight = [self.uploaded_weights[i] for i in idr_]
                for w, client_model in zip(weight, self.uploaded_models):
                    self.add_parameters(w, client_model)
                self.new_GM = copy.deepcopy(self.global_model)
                
                continue

            self.old_CM = copy.deepcopy(self.remaining_clients)
            
            # 得到新的CM，进行一步训练
            self.send_specific_models(self.new_GM)
            for i, client in enumerate(self.remaining_clients):
                client.set_parameters(remaining_clients_model[i])
                client.train_one_step()
            self.new_CM = copy.deepcopy(self.remaining_clients)
            
            # 开始校准
            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
        
        print(f"\n-------------After FedEraser-------------")
        print("\nEvaluate Eraser globel model")
        self.evaluate()
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
        
        
        return_global_model = copy.deepcopy(global_model_after_forget)
        
        return_global_model.load_state_dict(return_model_state)
        
        return return_global_model


    def retrain(self):
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.args.model)
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.remaining_clients
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            # self.save_client_model(i)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            # self.save_each_round_global_model(i)

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
        self.retrain_global_model = copy.deepcopy(self.global_model)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
            
    def MIA_metrics(self):
        attacker = self.build_MIA_attacker()
        print("\n-------------MIA evaluation against Standard FL-------------")
        (ACC_unlearn, PRE_unlearn) = self.MIA_attack(attacker, self.FL_global_model)
        
        print("\n-------------MIA evaluation against FL Unlearn-------------")
        (ACC_unlearn, PRE_unlearn) = self.MIA_attack(attacker, self.eraser_global_model)
        
        print("\n-------------MIA evaluation against FL Retrain-------------")
        (ACC_unlearn, PRE_unlearn) = self.MIA_attack(attacker, self.retrain_global_model)
        
        
    def build_MIA_attacker(self):
        from torch.nn.functional import softmax
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        shadow_model = self.FL_global_model
        n_class_dict = dict()
        n_class_dict['mnist'] = 10
        n_class_dict['cifar10'] = 10
        
        N_class = n_class_dict[self.dataset]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shadow_model.to(device)
            
        shadow_model.eval()
        
        # 得到使用的dataset的 [logits, 1]
        pred_4_mem = torch.zeros([1,N_class])
        pred_4_mem = pred_4_mem.to(device)
        with torch.no_grad():
            for ii in range(len(self.clients)):
                data_loader = self.clients[ii].load_train_data()
                
                for batch_idx, (data, target) in enumerate(data_loader):
                        data = data.to(device)
                        out = shadow_model(data)
                        pred_4_mem = torch.cat([pred_4_mem, out])
        pred_4_mem = pred_4_mem[1:,:]
        pred_4_mem = softmax(pred_4_mem,dim = 1)
        pred_4_mem = pred_4_mem.cpu()
        pred_4_mem = pred_4_mem.detach().numpy()
        
        # 得到未使用的dataset的 [logits, 0]
        pred_4_nonmem = torch.zeros([1,N_class])
        pred_4_nonmem = pred_4_nonmem.to(device)
        with torch.no_grad():
            for ii in range(len(self.clients)):
                data_loader = self.clients[ii].load_test_data()
                for batch, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = shadow_model(data)
                    pred_4_nonmem = torch.cat([pred_4_nonmem, out])
        pred_4_nonmem = pred_4_nonmem[1:,:]
        pred_4_nonmem = softmax(pred_4_nonmem,dim = 1)
        pred_4_nonmem = pred_4_nonmem.cpu()
        pred_4_nonmem = pred_4_nonmem.detach().numpy()
        
        
        #构建MIA 攻击模型 
        att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
        att_y = att_y.astype(np.int16)
        
        att_X = np.vstack((pred_4_mem, pred_4_nonmem))
        att_X.sort(axis=1)
        
        X_train,X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.1)
        
        attacker = XGBClassifier(n_estimators = 300,
                                n_jobs = -1,
                                    max_depth = 30,
                                objective = 'binary:logistic',
                                booster="gbtree",
                                # learning_rate=None,
                                # tree_method = 'gpu_hist',
                                scale_pos_weight = pred_4_nonmem.shape[0]/pred_4_mem.shape[0]
                                )
        
        attacker.fit(X_train, y_train)
        
        return attacker
        
    
    def MIA_attack(self, attacker, target_model):
        """使用在正常 FL 过程得到的 Global model, 测试遗忘程度

        Args:
            attacker (class): xgbboost 分类器
            target_model (nn.model): after unlearned global model

        Returns:
            (pre, rec)
        """
        from torch.nn.functional import softmax
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        n_class_dict = dict()
        n_class_dict['mnist'] = 10
        n_class_dict['cifar10'] = 10
        
        N_class = n_class_dict[self.dataset]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        target_model.eval()
        
        # 得到需要unlearn的client的 [logits, 1]
        unlearn_X = torch.zeros([1,N_class])
        unlearn_X = unlearn_X.to(device)
        with torch.no_grad():
            for ii in range(len(self.unlearn_clients)):
                data_loader = self.clients[ii].load_train_data()
                for batch, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = target_model(data)
                    unlearn_X = torch.cat([unlearn_X, out])
                        
        unlearn_X = unlearn_X[1:,:]
        unlearn_X = softmax(unlearn_X,dim = 1)
        unlearn_X = unlearn_X.cpu().detach().numpy()
        
        unlearn_X.sort(axis=1)
        unlearn_y = np.ones(unlearn_X.shape[0])
        unlearn_y = unlearn_y.astype(np.int16)
        
        N_unlearn_sample = len(unlearn_y)
        
        # 得到test的数据，标记为 [logits, 0]
        test_X = torch.zeros([1, N_class])
        test_X = test_X.to(device)
        with torch.no_grad():
            for ii in range(len(self.clients)):
                data_loader = self.clients[ii].load_test_data()
                for _, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = target_model(data)
                    test_X = torch.cat([test_X, out])
                
                    if(test_X.shape[0] > N_unlearn_sample):
                        break
                    
        test_X = test_X[1:N_unlearn_sample+1,:]
        test_X = softmax(test_X,dim = 1)
        test_X = test_X.cpu().detach().numpy()
        
        test_X.sort(axis=1)
        test_y = np.zeros(test_X.shape[0])
        test_y = test_y.astype(np.int16)
        
        # 理想状态应该是: unlearn的为全错，test的为全对
        XX = np.vstack((unlearn_X, test_X))
        YY = np.hstack((unlearn_y, test_y))
        
        pred_YY = attacker.predict(XX)
        # acc = accuracy_score( YY, pred_YY)
        pre = precision_score(YY, pred_YY, pos_label=1)
        rec = recall_score(YY, pred_YY, pos_label=1)
        # print("MIA Attacker accuracy = {:.4f}".format(acc))
        print("MIA Attacker precision = {:.4f}".format(pre))
        print("MIA Attacker recall = {:.4f}".format(rec))
        
        return (pre, rec)
    