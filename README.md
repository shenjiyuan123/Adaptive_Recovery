# Crab: Towards Efficient and Certified Recovery from Poisoning Attacks in Federated Learning

## About The Project
Official implementation of paper Crab. Crab can achieve efficient recovery from poisoning attacks through (i) selective storage of essential historical global models and clients' gradients rather than all historical information, and (ii) adaptive rollback to a global model that has not been significantly affected by the malicious clients rather than the initial model. 

## Generate the dataset
For example, if want to generate the MNIST, you can use
```
python generate_mnist.py iid - - # for iid and unbalanced scenario
# python generate_mnist.py iid balance - # for iid and balanced scenario
# python generate_mnist.py noniid - pat # for pathological noniid and unbalanced scenario
# python generate_mnist.py noniid - dir # for practical noniid and unbalanced scenario
```
Note: the file is 'add' mode, so notice to delete the original split files before generate the dataset.

## Federated Learning and Unlearning
We implement all the learning and unlearning function in the object of FedAvg, which inherit the Base Server object. 

One end-to-end running example contains Normal Learning, FedEraser, Retrain and MIA attack metrics. 
```
python FedMoss.py
```


## Acknowledge
This research is supported by the National Research Foundation, Singapore under its Strategic Capability Research Centres Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.


