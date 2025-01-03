import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the Simple Neural Network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to classify new data using the trained model
def classify_new_data(new_data, model_path):
    # Define the features used for classification
    feature_columns = [
        "area", "pleomorphism", "elongation", 
        "mean_intensity_DAPI", "total_intensity_DAPI"
    ]
    if not set(feature_columns).issubset(new_data.columns):
        raise ValueError(f"Missing required columns: {set(feature_columns) - set(new_data.columns)}")
    
    # Scale the numeric features
    scaler = StandardScaler()
    features = scaler.fit_transform(new_data[feature_columns])

    # Define the model dimensions
    input_dim = features.shape[1]
    hidden_dim1 = 64
    hidden_dim2 = 32
    output_dim = 2

    # Initialize and load the model
    model = SimpleNN(input_dim, hidden_dim1, hidden_dim2, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Convert features to a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)

    # Perform classification
    with torch.no_grad():
        outputs = model(features)
        _, predictions = torch.max(outputs, 1)

    # Add predictions to the original data
    new_data['lympho_classification'] = predictions.numpy()
    return new_data

# Main script for classification
if __name__ == "__main__":
    # File paths
    model_path = 'simple_nn_model.pth'
    new_data_path = '/nfs/cc-filer/home/sabulikailik/analysis/analysis_final/Processed_Images_Data21K_filtered_clustered.xlsx'

    # Load new data
    new_data = pd.read_excel(new_data_path)

    # Classify data
    try:
        classified_data = classify_new_data(new_data, model_path)

        # Save the classified data back to the same Excel file
        classified_data.to_excel(new_data_path, index=False)
        print(f'Classified data saved to {new_data_path}')
    except ValueError as e:
        print(f"Error: {e}")
