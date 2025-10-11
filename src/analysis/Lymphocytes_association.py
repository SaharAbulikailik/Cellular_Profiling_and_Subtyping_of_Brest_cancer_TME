import pandas as pd
import ast  # For safely evaluating strings containing lists

# Load the DataFrame
file_path = 'Processed_Images_Data_classified.xlsx'  # Update this path to your actual file
df = pd.read_excel(file_path)

# Define a function to count lymphocyte neighbors and calculate their frequencies
def calculate_neighbor_frequencies(row, df):
    # Extract neighbor IDs from the cellularity string
    neighbor_ids = df['neighbors']
    
    # Filter the DataFrame to include only rows with the same image_id and matching neighbor_ids
    neighbor_rows = df[(df['image_id'] == row['image_id']) & (df['nuclei_id'].isin(neighbor_ids))]
    lymphocyte_neighbors = neighbor_rows[neighbor_rows['lympho_pred'] == 1]

    # Calculate frequencies
    total_neighbors = len(neighbor_ids)
    lymphocyte_count = len(lymphocyte_neighbors)
    lymph_neighbor_frequency = lymphocyte_count / total_neighbors if total_neighbors > 0 else 0
    other_neighbor_frequency = (total_neighbors - lymphocyte_count) / total_neighbors if total_neighbors > 0 else 0

    return lymph_neighbor_frequency, other_neighbor_frequency

# Apply the function to each row to create the new columns
df['lymph_neighbor_frequency'], df['other_neighbor_frequency'] = zip(*df.apply(lambda row: calculate_neighbor_frequencies(row, df), axis=1))
df_final = df[['image_id', 'nuclei_id', 'lympho_pred', 'lymph_neighbor_frequency', 'other_neighbor_frequency']]

# Save the updated DataFrame to a new Excel file
output_file_path = 'final_merged1_classification_with_neighbors.xlsx'
df_final.to_excel(output_file_path, index=False)

# Display the updated DataFrame
print(df.head())
