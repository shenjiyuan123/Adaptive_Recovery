# normal 
# mnist
python FedMoss.py -verify -algo Retrain -unlearn 2 
python FedMoss.py -verify -algo FedEraser -unlearn 2 
python FedMoss.py -verify -algo FedRecover -unlearn 2
python FedMoss.py -verify -algo Crab -unlearn 2 

# fmnist
python FedMoss.py -data fmnist -verify -algo Retrain -unlearn 10
python FedMoss.py -data fmnist -verify -algo FedEraser -unlearn 10
python FedMoss.py -data fmnist -verify -algo FedRecover -unlearn 10
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10

# Cifar10
python FedMoss.py -data Cifar10 -verify -algo Retrain -unlearn 10 -lr 0.002
python FedMoss.py -data Cifar10 -verify -algo FedEraser -unlearn 10 -lr 0.002
python FedMoss.py -data Cifar10 -verify -algo FedRecover -unlearn 10 -lr 0.002
python FedMoss.py -data Cifar10 -verify -algo Crab -unlearn 10 -lr 0.002

# agnews
python FedMoss.py -verify -algo Retrain -unlearn 5 -data agnews -m Transformer -gr 25 --num_classes 4
python FedMoss.py -verify -algo FedEraser -unlearn 5 -data agnews -m Transformer -gr 25 --num_classes 4
python FedMoss.py -verify -algo FedRecover -unlearn 5 -data agnews -m Transformer -gr 25 --num_classes 4
python FedMoss.py -verify -algo Crab -unlearn 5 -data agnews -m Transformer -gr 25 --num_classes 4

# analysis
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.4 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.5 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.6 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.8 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.9 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 1.0 -gr 15 -backdoor

# include poisoning
# backdoor I
python FedMoss.py -verify -algo Retrain -data Cifar10 -unlearn 2 -backdoor -gr 15 --label_inject_mode Fix --tampered_label 3 -lr 0.002
python FedMoss.py -verify -algo FedEraser -data Cifar10 -unlearn 2 -backdoor -gr 15 -lr 0.002
python FedMoss.py -verify -algo FedRecover -data Cifar10 -unlearn 2 -backdoor -gr 15 -lr 0.002
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 2 -backdoor -gr 15 -lr 0.002

# backdoor II
python FedMoss.py -verify -algo Retrain -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo FedEraser -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo FedRecover -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002

# trim attack
python FedMoss.py -verify -algo Retrain -unlearn 5 -trim -data fmnist
python FedMoss.py -verify -algo FedEraser -unlearn 5 -trim -data fmnist
python FedMoss.py -verify -algo FedRecover -unlearn 5 -trim -data fmnist
python FedMoss.py -verify -algo Crab -unlearn 5 -trim -data fmnist

# Robust Aggregation
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 2 -backdoor -gr 15 -robust TrimmedMean
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 2 -backdoor -gr 15 -robust Median
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 2 -backdoor -gr 15 -robust Krum

python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Median
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust TrimmedMean
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Krum
