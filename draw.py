import pandas as pd
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(21, 7.4))

file_path = 'assets/cifar10_p10.csv'
data = pd.read_csv(file_path)

data_interpolated = data.copy()
data_interpolated['Crab'] = data['Crab'].interpolate()


######################################################################
ax1 = plt.subplot(1,3,1)

plt.plot(data_interpolated['Step'], data_interpolated['Retrain'], label='Retrain', marker='o', color='royalblue', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedRecover'], label='FedRecover', marker='v', color='crimson', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedEraser'], label='FedEraser', marker='^', color='forestgreen', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['Crab'], label='Crab', color='goldenrod', linewidth=3)

original_crab_data = data.dropna(subset=['Crab'])
plt.scatter(original_crab_data['Step'], original_crab_data['Crab'], color='goldenrod', marker='*', linewidths=2)

plt.xlabel('Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.ylim(1.0, 2.75)
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.grid(True, which='both', linestyle='-', linewidth=0.75, alpha=0.7)
plt.title('Target clients / Total clients = 10%')

######################################################################
plt.subplot(1,3,2)
file_path = 'assets/cifar10_p25.csv'
data = pd.read_csv(file_path)

data_interpolated = data.copy()
data_interpolated['Crab'] = data['Crab'].interpolate()

plt.plot(data_interpolated['Step'], data_interpolated['Retrain'], label='Retrain', marker='.', color='royalblue', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedRecover'], label='FedRecover', marker='v', color='crimson', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedEraser'], label='FedEraser', marker='o', color='forestgreen', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['Crab'], label='Crab', color='goldenrod', linewidth=3)

original_crab_data = data.dropna(subset=['Crab'])
plt.scatter(original_crab_data['Step'], original_crab_data['Crab'], color='goldenrod', marker='*', linewidths=2)

plt.xlabel('Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.ylim(1.0, 2.75)
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.grid(True, which='both', linestyle='-', linewidth=0.75, alpha=0.7)
plt.title('Target clients / Total clients = 25%')

######################################################################
plt.subplot(1,3,3)
file_path = 'assets/cifar10_p50.csv'
data = pd.read_csv(file_path)

data_interpolated = data.copy()
data_interpolated['Crab'] = data['Crab'].interpolate()

plt.plot(data_interpolated['Step'], data_interpolated['Retrain'], label='Retrain', marker='.', color='royalblue', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedRecover'], label='FedRecover', marker='v', color='crimson', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['FedEraser'], label='FedEraser', marker='o', color='forestgreen', linewidth=2, markersize=6)
plt.plot(data_interpolated['Step'], data_interpolated['Crab'], label='Crab', color='goldenrod', linewidth=3)

original_crab_data = data.dropna(subset=['Crab'])
plt.scatter(original_crab_data['Step'], original_crab_data['Crab'], color='goldenrod', marker='*', linewidths=2)

plt.xlabel('Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.ylim(1.0, 2.75)
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.grid(True, which='both', linestyle='-', linewidth=0.75, alpha=0.7)
plt.title('Target clients / Total clients = 50%')

######################################################################
plt.tight_layout(pad=3.0)
plt.suptitle("CIFAR10", fontsize=20)
# plt.show()
plt.savefig('assets/cifar10_loss_vary2.pdf')

