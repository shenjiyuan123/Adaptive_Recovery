# normal 
# mnist
# python FedMoss.py -verify -algo Retrain -unlearn 2 
# python FedMoss.py -verify -algo FedEraser -unlearn 2 
# python FedMoss.py -verify -algo FedRecover -unlearn 2
# python FedMoss.py -verify -algo Crab -unlearn 2 

# fmnist
# python FedMoss.py -data fmnist -verify -algo Retrain -unlearn 10
# python FedMoss.py -data fmnist -verify -algo FedEraser -unlearn 10
# python FedMoss.py -data fmnist -verify -algo FedRecover -unlearn 10
# python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10

# Cifar10
# python FedMoss.py -data Cifar10 -verify -algo Retrain -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo FedEraser -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo FedRecover -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo Crab -unlearn 10

# backdoor 
# python FedMoss.py -verify -algo Retrain -data Cifar10 -unlearn 2 -backdoor -gr 15
# python FedMoss.py -verify -algo FedEraser -data Cifar10 -unlearn 2 -backdoor -gr 15
# python FedMoss.py -verify -algo FedRecover -data Cifar10 -unlearn 2 -backdoor -gr 15
# python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 2 -backdoor -gr 15

# trim 
# python FedMoss.py -verify -algo Retrain -unlearn 5 -trim -data fmnist
# python FedMoss.py -verify -algo FedEraser -unlearn 5 -trim -data fmnist
# python FedMoss.py -verify -algo FedRecover -unlearn 5 -trim -data fmnist
# python FedMoss.py -verify -algo Crab -unlearn 5 -trim -data fmnist

# python FedMoss.py -verify -algo Retrain -unlearn 5 -trim -data Cifar10
# python FedMoss.py -verify -algo FedEraser -unlearn 5 -trim -data Cifar10
# python FedMoss.py -verify -algo FedRecover -unlearn 5 -trim -data Cifar10
# python FedMoss.py -verify -algo Crab -unlearn 5 -trim -data Cifar10

# analysis
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.4 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.5 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.6 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.8 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 0.9 -gr 15 -backdoor
python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10 -Xclients 1.0 -gr 15 -backdoor

