# Crab: Towards Efficient and Certified Recovery from Poisoning Attacks in Federated Learning

## About The Project
Official implementation of paper Crab. Crab can achieve efficient recovery from poisoning attacks through (i) selective storage of essential historical global models and clients' gradients rather than all historical information, and (ii) adaptive rollback to a global model that has not been significantly affected by the malicious clients rather than the initial model. 

## Generate the dataset
For example, if want to generate the MNIST, you can use
```
python generate_mnist.py iid - -                # for iid and unbalanced scenario
# python generate_mnist.py iid balance -        # for iid and balanced scenario
# python generate_mnist.py noniid - pat         # for pathological noniid and unbalanced scenario   
# python generate_mnist.py noniid - dir         # for practical noniid and unbalanced scenario
```
Note: the split file of train/test is 'add' mode, so notice to delete the original split files if you want to re-split the dataset.

## How to start the end-to-end recovery
For now, the support recovery algorithms includes **Retrain, FedEraser, FedRecover and Crab**. 

For example, to execute the `Crab` recovery simulation, run the following commands:
```python
python FedMoss.py       --dataset mnist                 \
                        --global_rounds 20              \
                        --local_epochs 5                \
                        --algorithm Crab                \    
                        --verify_unlearn True           \    
                        --num_clients 20                \    
                        --unlearn_clients_number 10     \     
```

- `dataset`: The name of experiment dataset.
- `global_rounds`: The communication rounds between clients and server.
- `local_epochs`: The training epochs of each clients. 
- `algorithm`: The recovery algorithm, can choose **Retrain, FedEraser, FedRecover and Crab**.
- `verify_unlearn`: Whether use the MIA or backdoor to verify the unlearning effectiveness.
- `num_clients`: The total number of clients.
- `unlearn_clients_number`: The targeted client number.




## Acknowledge
This research is supported by the National Research Foundation, Singapore under its Strategic Capability Research Centres Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.


