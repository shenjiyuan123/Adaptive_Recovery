import torch
import torch.nn as nn
import os
import numpy as np
import h5py
import copy
import time
import random
from pprint import pprint
from torch.nn.functional import softmax
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from MIA_utils import ShadowDataset
from torch.utils.data import DataLoader

from clientBase import clientAVG
from dataset_utils import read_client_data



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
        self.initial_model = copy.deepcopy(args.model)
        # print("initial:", self.initial_model.state_dict()['base.conv1.0.weight'][0])
        self.global_model = copy.deepcopy(args.model)
        self.total_clients = len(os.listdir('data/mnist/test'))
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
        self.backdoor_attack = args.backdoor_attack

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
        
        self.remaining_clients = []

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
            
    def receive_retrained_models(self, remaining_clients):
        assert (len(remaining_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in remaining_clients:
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
        
            
    def save_client_model(self, epoch):
        model_path = os.path.join("clients_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
        torch.save(self.selected_clients, model_path)

    def load_client_model(self, epoch):
        model_path = os.path.join("clients_models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
        assert (os.path.exists(model_path))
        client_models = torch.load(model_path)
        assert (len(client_models) == self.num_clients)
        return client_models


    def save_global_model(self):
        model_path = os.path.join("server_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
        
    def save_each_round_global_model(self, epoch):
        model_path = os.path.join("server_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("server_models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)
            

    def model_exists(self):
        model_path = os.path.join("server_models", self.dataset)
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
    
    
    def read_all_testset(self):
        raise NotImplementedError
        from dataset_utils import read_all_test_data
        import os
        from torch.utils.data import DataLoader
        range_idx = len(os.listdir("data/mnist/test"))
        test_data = read_all_test_data(self.dataset, range_idx)
        return DataLoader(test_data, 64, drop_last=False, shuffle=True)

    def server_metrics(self):
        import dataset_utils
        from torch.utils.data import DataLoader
        from sklearn.preprocessing import label_binarize
        from sklearn import metrics
        testdata = dataset_utils.read_all_test_data(self.dataset, self.total_clients)
        testloader = DataLoader(testdata, self.batch_size, drop_last=False, shuffle=True)
        self.global_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        acc = test_acc / test_num
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        print("Average Test Accurancy: {:.4f}".format(acc))
        print("Average Test AUC: {:.4f}".format(auc))
        return acc, acc

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        if not self.remaining_clients:
            for c in self.clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
            ids = [c.id for c in self.clients]
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
        if not self.remaining_clients:
            for c in self.clients:
                cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl*1.0)

            ids = [c.id for c in self.clients]
        else:
            for c in self.remaining_clients:
                cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl*1.0)

            ids = [c.id for c in self.remaining_clients]

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
    ''' 
    Unlearning base functions: Select target clients, MIA attack,
    '''
    
    def select_unlearned_clients(self):
        self.selected_clients = self.select_clients()
        
        id_selected_clients = [c.id for c in self.selected_clients] 
        
        idx_ = random.sample(id_selected_clients, self.unlearn_clients_number)
        # idx_ = [9]
        idr_ = [i for i in id_selected_clients if i not in idx_]
        self.idr_ = idr_
        self.idx_ = idx_
        # idr_ = [0,1,2,3,4,5,6,7,8]
        self.unlearn_clients = [c for c in self.selected_clients if c.id in idx_]
        self.remaining_clients = [c for c in self.selected_clients if c.id in idr_]
        # print(self.unlearn_clients, self.remaining_clients)
        print(f"Target clients id: {idx_} \nRemaining clients id: {idr_}.")
    
    def MIA_metrics(self):
        # self.FL_global_model = torch.load('models/10_2/FedAvg_server.pt')
        # self.retrain_global_model = torch.load('models/9_2/FedAvg_server.pt')
        
        print(self.FL_global_model.state_dict()['base.conv1.0.weight'][-1] - self.retrain_global_model.state_dict()['base.conv1.0.weight'][-1])
        print(self.FL_global_model.state_dict()['base.conv1.0.weight'][-1] - self.eraser_global_model.state_dict()['base.conv1.0.weight'][-1])
        
        del self.remaining_clients
        
        attacker = self.build_MIA_attacker()
        # attacker = self.train_attack()
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
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
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
        # print("self remaining clients", self.remaining_clients)
        with torch.no_grad():
            for client in self.clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    data_loader = client.load_train_data(create_trigger=True)
                else:
                    data_loader = client.load_train_data()
                for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = shadow_model(data)
                    pred_4_mem = torch.cat([pred_4_mem, out])
            # for client in self.remaining_clients:
            #     data_loader = client.load_train_data()
            #     for batch_idx, (data, target) in enumerate(data_loader):
            #         data = data.to(device)
            #         out = shadow_model(data)
            #         pred_4_mem = torch.cat([pred_4_mem, out])
        pred_4_mem = pred_4_mem[1:,:]
        pred_4_mem = softmax(pred_4_mem,dim = 1)
        pred_4_mem = pred_4_mem.cpu()
        pred_4_mem = pred_4_mem.detach().numpy()
        unlearn_data_nums = pred_4_mem.shape[0]
        
        # 得到未使用的dataset的 [logits, 0]
        import dataset_utils
        testset = dataset_utils.read_all_test_data(self.dataset, self.total_clients)
        testloader = torch.utils.data.DataLoader(testset, self.batch_size, drop_last=False, shuffle=True)
        
        pred_4_nonmem = torch.zeros([1,N_class])
        pred_4_nonmem = pred_4_nonmem.to(device)
        with torch.no_grad():
            for batch, (data, target) in enumerate(testloader):
                data = data.to(device)
                out = shadow_model(data)
                pred_4_nonmem = torch.cat([pred_4_nonmem, out])
        pred_4_nonmem = pred_4_nonmem[1:unlearn_data_nums+1,:]
        pred_4_nonmem = softmax(pred_4_nonmem,dim = 1)
        pred_4_nonmem = pred_4_nonmem.cpu()
        pred_4_nonmem = pred_4_nonmem.detach().numpy()
        
        
        #构建MIA 攻击模型 
        att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
        att_y = att_y.astype(np.int16)
        
        att_X = np.vstack((pred_4_mem, pred_4_nonmem))
        # att_X.sort(axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.1)
        print(f"Training samples size: {pred_4_mem.shape[0]}, Test samples size: {pred_4_nonmem.shape[0]}")
        
        attacker = XGBClassifier(n_estimators = 500,
                                n_jobs = -1,
                                max_depth = 30,
                                objective = 'binary:logistic',
                                booster= "gbtree",
                                tree_method = 'gpu_hist',
                                device = self.device,
                                scale_pos_weight = pred_4_nonmem.shape[0]/pred_4_mem.shape[0]
                                )
        
        attacker.fit(X_train, y_train)
        pred_YY = attacker.predict(X_test)
        acc = accuracy_score(y_test, pred_YY)
        pre = precision_score(y_test, pred_YY, pos_label=1)
        rec = recall_score(y_test, pred_YY, pos_label=1)
        print("\n-------------Test Membership Inference Capcity-------------")
        pred = attacker.predict(X_train)
        print("Test MIA Attacker train accuracy = {:.4f}".format(accuracy_score(y_train, pred)))
        print("Test MIA Attacker test accuracy = {:.4f}".format(acc))
        
        return attacker
         
        
    
    def MIA_attack(self, attacker, target_model, T=0.9):
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
        print(len(self.unlearn_clients))
        with torch.no_grad():
            for ii in range(len(self.unlearn_clients)):
                if self.backdoor_attack:
                    data_loader = self.unlearn_clients[ii].load_train_data(create_trigger=True)
                else:
                    data_loader = self.unlearn_clients[ii].load_train_data()
                for batch, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = target_model(data)
                    unlearn_X = torch.cat([unlearn_X, out])
                        
        unlearn_X = unlearn_X[1:,:]
        unlearn_X = softmax(unlearn_X,dim = 1)
        unlearn_X = unlearn_X.cpu().detach().numpy()
        print(unlearn_X.shape)
        
        # unlearn_X.sort(axis=1)
        unlearn_y = np.ones(unlearn_X.shape[0])
        unlearn_y = unlearn_y.astype(np.int16)
        
        N_unlearn_sample = len(unlearn_y)
        
        # 得到test的数据，标记为 [logits, 0]
        test_X = torch.zeros([1, N_class])
        test_X = test_X.to(device)
        import dataset_utils
        testset = dataset_utils.read_all_test_data(self.dataset, self.total_clients)
        testloader = torch.utils.data.DataLoader(testset, self.batch_size, drop_last=False, shuffle=True)
        with torch.no_grad():
            for _, (data, target) in enumerate(testloader):
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
        

        # 理想状态应该是: unlearn的为全错，test的为全对. 
        XX = np.vstack((unlearn_X))
        YY = np.hstack((unlearn_y))
        
        pred_YY = attacker.predict(XX)
        acc = accuracy_score(YY, pred_YY)
        pre = precision_score(YY, pred_YY, pos_label=1)
        rec = recall_score(YY, pred_YY, pos_label=1)
        print("MIA Attacker accuracy = {:.4f}".format(acc))
        print("MIA Attacker precision = {:.4f}".format(pre))
        print("MIA Attacker recall = {:.4f}".format(rec))
        
        return (pre, rec)
    
    
    def retrain(self):
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.initial_model)
        for client in self.remaining_clients:
            client.set_parameters(self.global_model)
        print(self.global_model.state_dict()['base.conv1.0.weight'][0])

        for i in range(self.global_rounds+1):
            s_t = time.time()

            assert (len(self.clients) > 0)
            for client in self.remaining_clients:
                start_time = time.time()
                
                client.set_parameters(self.global_model)
                
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
                

            if i%self.eval_gap == 0:
                print(f"\n-------------Retrain Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                # self.server_metrics()
                
            # print(self.remaining_clients, len(self.remaining_clients), len(self.unlearn_clients))
            for client in self.remaining_clients:
                client.train()

            self.receive_retrained_models(self.remaining_clients)
            self.aggregate_parameters()
            print("retrain ***:::", self.global_model.state_dict()['base.conv1.0.weight'][0])

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
        self.server_metrics()
        self.retrain_global_model = copy.deepcopy(self.global_model)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
  

    ################################################################################################################   
    ''' 
    An alternative to construct the MIA model by using MLP
    '''
    def train_attack(self):
        
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
        # print("self remaining clients", self.remaining_clients)
        with torch.no_grad():
            for client in self.clients:
                data_loader = client.load_train_data()
                for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = shadow_model(data)
                    pred_4_mem = torch.cat([pred_4_mem, out])
            # for client in self.remaining_clients:
            #     data_loader = client.load_train_data()
            #     for batch_idx, (data, target) in enumerate(data_loader):
            #         data = data.to(device)
            #         out = shadow_model(data)
            #         pred_4_mem = torch.cat([pred_4_mem, out])
        pred_4_mem = pred_4_mem[1:,:]
        pred_4_mem = softmax(pred_4_mem,dim = 1)
        pred_4_mem = pred_4_mem.cpu()
        pred_4_mem = pred_4_mem.detach().numpy()
        unlearn_data_nums = pred_4_mem.shape[0]
        
        # 得到未使用的dataset的 [logits, 0]
        import dataset_utils
        testset = dataset_utils.read_all_test_data(self.dataset, self.total_clients)
        testloader = torch.utils.data.DataLoader(testset, self.batch_size, drop_last=False, shuffle=True)
        
        pred_4_nonmem = torch.zeros([1,N_class])
        pred_4_nonmem = pred_4_nonmem.to(device)
        with torch.no_grad():
            for batch, (data, target) in enumerate(testloader):
                data = data.to(device)
                out = shadow_model(data)
                pred_4_nonmem = torch.cat([pred_4_nonmem, out])
        pred_4_nonmem = pred_4_nonmem[1:unlearn_data_nums+1,:]
        pred_4_nonmem = softmax(pred_4_nonmem,dim = 1)
        pred_4_nonmem = pred_4_nonmem.cpu()
        pred_4_nonmem = pred_4_nonmem.detach().numpy()
        
        
        #构建MIA 攻击模型 
        att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
        att_y = att_y.astype(np.int16)
        
        att_X = np.vstack((pred_4_mem, pred_4_nonmem))
        # att_X.sort(axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.1)
        print(pred_4_mem.shape[0], pred_4_nonmem.shape[0])
        # Create dataset instances
        train_dataset = ShadowDataset(X_train, y_train)
        test_dataset = ShadowDataset(X_test, y_test)

        # Now let's move on to creating the DataLoader.
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        from trainmodel.models import MLP
        attacker = MLP(in_features=out.shape[1], num_classes=2, hidden_dim=64).to(self.device)
        optimiz = torch.optim.Adam(params=attacker.parameters(), lr=self.learning_rate)
        celoss = nn.CrossEntropyLoss()
        
        attacker.train()
        for step in range(50):
            for i, (x, y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output =attacker(x)
                loss = celoss(output, y)
                optimiz.zero_grad()
                loss.backward()
                optimiz.step()
            if step%5 == 0:
                print(f"In epoch {step}, the loss of the attacker is {loss}.")
                
        test_acc = test_num = 0
        
        attacker.eval() 
        with torch.no_grad():
            for x, y in test_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = attacker(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        print("Test MIA Attacker test accuracy = {:.4f}".format(test_acc/test_num))
        
        return attacker
            
            

    