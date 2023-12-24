# normal 
# mnist
# python FedMoss.py -verify -algo Retrain -unlearn 2 
# python FedMoss.py -verify -algo FedEraser -unlearn 2 
# python FedMoss.py -verify -algo FedRecover -unlearn 2
# python FedMoss.py -verify -algo Crab -unlearn 2 

# Cifar10
# python FedMoss.py -data Cifar10 -verify -algo Retrain -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo FedEraser -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo FedRecover -unlearn 10
# python FedMoss.py -data Cifar10 -verify -algo Crab -unlearn 10

# fmnist
# python FedMoss.py -data fmnist -verify -algo Retrain -unlearn 10
# python FedMoss.py -data fmnist -verify -algo FedEraser -unlearn 10
# python FedMoss.py -data fmnist -verify -algo FedRecover -unlearn 10
# python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10

# agnews
# python FedMoss.py -data agnews -m TextCNN -gr 20 --num_classes 4 -verify -algo Retrain -unlearn 10

# backdoor 
python FedMoss.py -verify -algo Retrain -data Cifar10 -unlearn 10 -backdoor
python FedMoss.py -verify -algo FedEraser -data Cifar10 -unlearn 10 -backdoor
python FedMoss.py -verify -algo FedRecover -data Cifar10 -unlearn 10 -backdoor
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 10 -backdoor

# trim 
# python FedMoss.py -verify -algo Retrain -unlearn 5 -trim
# python FedMoss.py -verify -algo FedEraser -unlearn 5 -trim
# python FedMoss.py -verify -algo FedRecover -unlearn 5 -trim
# python FedMoss.py -verify -algo Crab -unlearn 5 -trim
