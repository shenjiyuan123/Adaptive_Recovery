#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import torch.nn as nn
import wandb

from serverEraser import FedEraser
from serverCrab import Crab
from serverFedRecover import FedRecover

from trainmodel.models import *

from trainmodel.bilstm import *
from trainmodel.resnet import *
from trainmodel.alexnet import *
from trainmodel.mobilenet_v2 import *
from trainmodel.transformer import *


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32

def run(args):

    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr": # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn": # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn": # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)
            
        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            
        else:
            raise NotImplementedError

        print(args.model)
        
        wandb.init(sync_tensorboard=False,
               project="FedMoss",
               job_type="CleanRepo",
               config=args,
               )

        # select algorithm
        if args.algorithm == "FedEraser":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedEraser(args, i)
            
            server.select_unlearned_clients()
            server.train()
            server.unlearning()
            # server.retrain()
            
        elif args.algorithm == "FedRecover":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRecover(args, i)
            
            server.select_unlearned_clients()
            server.train()
            # server.retrain()
            server.recover()
        
        elif args.algorithm == "Crab":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Crab(args, i)
            
            server.select_unlearned_clients()
            server.train_with_select()
            server.adaptive_recover()
            # server.retrain()
            
        elif args.algorithm == "Retrain":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedEraser(args, i)
            
            server.select_unlearned_clients()
            server.train()
            server.retrain()
            
        else:
            raise NotImplementedError

        if args.verify_unlearn:
            server.MIA_metrics()


        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    print("All done!")



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('-go', "--goal", type=str, default="Federated Unlearning Experiments", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=256)  # -> 256
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=40)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    
    # unlearning settings
    parser.add_argument('-algo', "--algorithm", type=str, default="FedEraser", choices=["Retrain", "FedEraser", "FedRecover", "Crab"],
                        help="How to unlearn the target clients")
    parser.add_argument('-verify', "--verify_unlearn", action='store_true',
                        help="Whether use the MIA to verify the unlearn effectiveness")
    parser.add_argument('-Prounds', "--select_round_ratio", type=float, default=0.6)
    parser.add_argument('-Xclients', "--select_client_ratio", type=float, default=0.7)
    
    # robust aggregation schemes
    parser.add_argument('-robust', "--robust_aggregation_schemes", type=str, default="FedAvg",
                        choices=["TrimmedMean", "Median", "Krum"], help="The aggregation schemes using when calculating the server parameters")
    parser.add_argument("--trimmed_clients_num", type=int, default=2, 
                        help="The number of clients will be trimmed. Calculated by each dimensions.")
    
    # Crab settings
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-unlearn', "--unlearn_clients_number", type=int, default=5,
                        help="Total number of unlearn clients")
    
    # attack setting
    # backdoor
    parser.add_argument('-backdoor', '--backdoor_attack', action='store_true', 
                    help="Whether to inject backdoor attack towards the target clients")
    parser.add_argument('--trigger_size', type=int, default=4,
                        help="Size of injected trigger")
    parser.add_argument('--label_inject_mode', type=str, default="Fix", choices=["Fix", "Random", "Exclusive"], 
                        help="Random: asign tampered label randomly to each original label. Exclusive: perturb all the data with specific label and trigger.")
    parser.add_argument('-clamp', '--clamp_to_little_range', action='store_true', 
                        help="whether to further clamp the updated parameters to little range based on malicous and benign params so that can circumvent defenses. From paper 'A Little Is Enough: Circumventing Defenses For Distributed Learning'.")
    parser.add_argument('--tampered_label', type=int, default=2,
                        help="Tamper label that corresponds to the sample injected the backdoor trigger. Must set '--label_inject_mode' to Fix")
    # trim
    parser.add_argument('-trim', '--trim_attack', action='store_true', 
                    help="Whether to execute trim attack towards the target clients")
    parser.add_argument('--trim_percentage', type=int, default=20,
                        help="Percentage of execute trim attack towards the target clients")
    
    # trivial
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    
    print(args.__dict__)
    
    print("=" * 50)

    # print("Algorithm: {}".format(args.algorithm))
    # print("Local batch size: {}".format(args.batch_size))
    # print("Local steps: {}".format(args.local_epochs))
    # print("Local learing rate: {}".format(args.local_learning_rate))
    # print("Local learing rate decay: {}".format(args.learning_rate_decay))
    # if args.learning_rate_decay:
    #     print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    # print("Total number of clients: {}".format(args.num_clients))
    # print("Clients join in each round: {}".format(args.join_ratio))
    # print("Clients randomly join: {}".format(args.random_join_ratio))
    # print("Client drop rate: {}".format(args.client_drop_rate))
    # print("Client select regarding time: {}".format(args.time_select))
    # if args.time_select:
    #     print("Time threthold: {}".format(args.time_threthold))
    # print("Running times: {}".format(args.times))
    # print("Dataset: {}".format(args.dataset))
    # print("Number of classes: {}".format(args.num_classes))
    # print("Backbone: {}".format(args.model))
    # print("Using device: {}".format(args.device))
    # print("Using DP: {}".format(args.privacy))
    # if args.privacy:
    #     print("Sigma for DP: {}".format(args.dp_sigma))
    # print("Auto break: {}".format(args.auto_break))
    # if not args.auto_break:
    #     print("Global rounds: {}".format(args.global_rounds))
    # if args.device == "cuda":
    #     print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # print("DLG attack: {}".format(args.dlg_eval))
    # if args.dlg_eval:
    #     print("DLG attack round gap: {}".format(args.dlg_gap))
    # print("Total number of new clients: {}".format(args.num_new_clients))
    # print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    # print("=" * 50)

    run(args)

    