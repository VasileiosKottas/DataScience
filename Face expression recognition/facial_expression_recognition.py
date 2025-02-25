import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load the dataset and encode target labels
data = pd.read_excel("cohn-kanade-rev_new.xls")
label_encoder = LabelEncoder()
data['Expression'] = label_encoder.fit_transform(data['Expression'])

# Separate features and labels, and remove rows with NaN values
X = data.iloc[:, :-1]
y = data['Expression']
X_numeric = X.select_dtypes(include=[np.number]).dropna()
y = y[X_numeric.index]

# Feature selection to select top features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_numeric, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare the data for PyTorch by creating a custom dataset class
class FaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create training and testing datasets and dataloaders
train_dataset = FaceDataset(X_train, y_train)
test_dataset = FaceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the neural network model
class ExpressionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ExpressionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = ExpressionClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network using a loop with a progress bar
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# Evaluate the neural network on the test data
model.eval()
y_pred = []
with torch.no_grad():
    for features, _ in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Visualize the performance using a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)

# Perform KMeans clustering on the original features
kmeans = KMeans(n_clusters=num_classes, random_state=42)
clusters = kmeans.fit_predict(X_selected)
cluster_accuracy = accuracy_score(y, clusters)
silhouette = silhouette_score(X_selected, clusters)
print(f"Clustering Accuracy (Original Features): {cluster_accuracy * 100:.2f}%")
print(f"Silhouette Score: {silhouette:.2f}")

# Visualize the KMeans clustering results
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=clusters, cmap='viridis', marker='o', alpha=0.6, label='Clusters')
plt.title("KMeans Clustering (Original Features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.legend()
plt.savefig("kmeans_clustering_original.png")
plt.show()

# Visualize the true labels in PCA-reduced space
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', marker='x', alpha=0.6, label='True Labels')
plt.title("True Labels (PCA-Reduced Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Expression Label")
plt.legend()
plt.savefig("true_labels.png")
plt.show()