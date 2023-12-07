import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Assuming att_X and att_y are numpy arrays that you have
# For demonstration, let's create dummy data similar to att_X and att_y
# Example:
# att_X = np.random.rand(1000, 28 * 28)  # Example feature data (1000 samples, 28x28 flattened images)
# att_y = np.random.randint(0, 10, 1000)  # Example labels (1000 samples, labels from 0 to 9)

class ShadowDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        :param features: Input features (numpy array).
        :param labels: Corresponding labels (numpy array).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Fetch the sample at the given index.
        :param idx: Index of the sample to fetch.
        :return: A tuple (feature, label).
        """
        return self.features[idx], self.labels[idx]

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(att_X, att_y, test_size=0.1)

# Create dataset instances
# train_dataset = CustomDataset(X_train, y_train)
# test_dataset = CustomDataset(X_test, y_test)

# Now let's move on to creating the DataLoader.
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Continue with defining a simple neural network, training and testing it on the provided dataset.
# Please provide the actual datasets or let me know if you want me to continue with dummy data.




