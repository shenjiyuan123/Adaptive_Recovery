python FedMoss.py -verify -algo Retrain -data mnist -unlearn 2 -backdoor -clamp -gr 20 
python FedMoss.py -verify -algo FedEraser -data mnist -unlearn 2 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo FedRecover -data mnist -unlearn 2 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo Crab -data mnist -unlearn 2 -backdoor -clamp -gr 20

python FedMoss.py -verify -algo Retrain -data mnist -unlearn 5 -backdoor -clamp -gr 20 
python FedMoss.py -verify -algo FedEraser -data mnist -unlearn 5 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo FedRecover -data mnist -unlearn 5 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo Crab -data mnist -unlearn 5 -backdoor -clamp -gr 20

python FedMoss.py -verify -algo Retrain -data fmnist -unlearn 10 -backdoor -clamp -gr 20 
python FedMoss.py -verify -algo FedEraser -data fmnist -unlearn 10 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo FedRecover -data fmnist -unlearn 10 -backdoor -clamp -gr 20
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 10 -backdoor -clamp -gr 20

python FedMoss.py -verify -algo Retrain -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo FedEraser -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo FedRecover -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 10 -backdoor -clamp -gr 25 -lr 0.002

python FedMoss.py -verify -algo Retrain -data agnews -m Transformer --num_classes 4 -unlearn 10 -backdoor -clamp -gr 25 
python FedMoss.py -verify -algo FedEraser -data agnews -m Transformer --num_classes 4 -unlearn 10 -backdoor -clamp -gr 25
python FedMoss.py -verify -algo FedRecover -data agnews -m Transformer --num_classes 4 -unlearn 10 -backdoor -clamp -gr 25
python FedMoss.py -verify -algo Crab -data agnews -m Transformer --num_classes 4 -unlearn 10 -backdoor -clamp -gr 25

########################################################################################################

python FedMoss.py -verify -algo Crab -data mnist -unlearn 5 -backdoor -clamp -gr 20 -robust Median
python FedMoss.py -verify -algo Crab -data mnist -unlearn 5 -backdoor -clamp -gr 20 -robust TrimmedMean
python FedMoss.py -verify -algo Crab -data mnist -unlearn 5 -backdoor -clamp -gr 20 -robust Krum

python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Median
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust TrimmedMean
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Krum

python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 5 -backdoor -clamp -gr 20 -robust Median -lr 0.002
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 5 -backdoor -clamp -gr 20 -robust TrimmedMean -lr 0.002
python FedMoss.py -verify -algo Crab -data Cifar10 -unlearn 5 -backdoor -clamp -gr 20 -robust Krum -lr 0.002

python FedMoss.py -verify -algo Crab -data agnews -m Transformer --num_classes 4 -unlearn 5 -backdoor -clamp -gr 25 -robust Median
python FedMoss.py -verify -algo Crab -data agnews -m Transformer --num_classes 4 -unlearn 5 -backdoor -clamp -gr 25 -robust TrimmedMean
python FedMoss.py -verify -algo Crab -data agnews -m Transformer --num_classes 4 -unlearn 5 -backdoor -clamp -gr 25 -robust Krum