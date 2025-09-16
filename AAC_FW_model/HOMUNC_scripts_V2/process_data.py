# %%
"""
Create mat files containing d_delta correctly formatted for corresponding trial congruency.
If choice == "fora": p_del = current_weather_p_success - alternative_weather_p_success
If choice == "contra": p_del = alternative_weather_p_success - current_weather_p_success
This script reads a CSV file, processes the data to calculate the p_del column,
and exports the data for each subject into separate .mat files ready to be used by SPM.
"""
import pandas as pd
import numpy as np
import os
from scipy.io import savemat

def process_csv_and_export():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('datall_cat.csv')
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Create the p_del column
    print("Calculating p_del values...")
    
    def calculate_p_del(row):
        choice = row['x11_choice']
        current_p_success = row['x14_p_foraging_gain']
        weather_1_p_gain = row['x59_weather_1_p_gain']
        weather_2_p_gain = row['x60_weather_2_p_gain']
        
        # Determine which weather alternative is different from current
        if weather_1_p_gain != current_p_success:
            alternative_p_success = weather_1_p_gain
        else:
            alternative_p_success = weather_2_p_gain
        
        # Calculate p_del
        # Calculate p_del based on choice
        if choice == 1:
            p_del = current_p_success - alternative_p_success
        elif choice == 0:
            p_del = alternative_p_success - current_p_success
        else:
            p_del = 0  # Default case for unexpected values
            
        return p_del
    
    # Apply the calculation to create p_del column
    df['p_del'] = df.apply(calculate_p_del, axis=1)
    
    print("p_del column created successfully!")
    print(f"p_del statistics:")
    print(f"  Mean: {df['p_del'].mean():.4f}")
    print(f"  Std: {df['p_del'].std():.4f}")
    print(f"  Min: {df['p_del'].min():.4f}")
    print(f"  Max: {df['p_del'].max():.4f}")
    
    # Create output directory
    output_dir = 'beh_v12_agg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get unique subject IDs
    unique_subjects = df['x1_id'].unique()
    print(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}")
    
    # Export data per subject as .mat files
    print("Exporting data per subject...")
    for subject_id in unique_subjects:
        # Filter data for this subject
        subject_data = df[df['x1_id'] == subject_id]
        
        # Remove 'x' prefix from column names
        clean_headers = [col.replace('x', '', 1) if col.startswith('x') else col for col in subject_data.columns]
        
        # Create the data matrix (all values) and headers
        # Remove the first column (index) and create the data matrix
        subject_data_no_index = subject_data.iloc[:, 1:]  # Skip first column
        clean_headers_no_index = clean_headers[1:]  # Skip first header
        
        mat_data = {
            'x': subject_data_no_index.values,  # Matrix with all values (no index column)
            'headers': clean_headers_no_index  # Column names with 'x' prefix removed (no index)
        }

        # Create filename
        filename = f"beh_v12_sub{subject_id}.mat"
        filepath = os.path.join(output_dir, filename)
        
        # Save as .mat file
        savemat(filepath, mat_data)
        print(f"  Exported subject {subject_id}: {len(subject_data)} rows -> {filename}")
    
    # Print MATLAB-formatted column names
    print("\n" + "="*80)
    print("MATLAB HEADER FORMAT FOR NEW COLUMNS:")
    print("="*80)
    
    # Get all column names with 'x' prefix removed
    clean_columns = [col.replace('x', '', 1) if col.startswith('x') else col for col in df.columns]
    
    print("All column names in MATLAB cell array format:")
    print("{'" + "', '".join(clean_columns) + "'}")
    
    print("\nNew column added:")
    # Find all columns starting from x57_ (with x removed)
    x57_onwards = [col.replace('x', '', 1) if col.startswith('x') else col 
                   for col in df.columns 
                   if col.startswith('x57_') or 
                   (col.startswith('x') and col.split('_')[0][1:].isdigit() and 
                    int(col.split('_')[0][1:]) >= 57) or col == 'p_del']
    formatted_columns = ["{'" + col + "'}" for col in x57_onwards]
    print(", ".join(formatted_columns))
    print('Number of new columns:', len(x57_onwards))
    
    print("\nColumn descriptions for MATLAB:")
    print("% p_del: Probability current weather - other weather")
    # Print some sample data for verification
    print("\n" + "="*80)
    print("SAMPLE DATA VERIFICATION:")
    print("="*80)
    sample_data = df[['x1_id', 'x11_choice', 'x6_continuous_energy_trial_start', 'x14_p_foraging_gain', 'x59_weather_1_p_gain', 'x60_weather_2_p_gain', 'p_del']].head(10)
    print(sample_data.to_string(index=False))
    
    print(f"\nProcessing completed! Files saved in: {output_dir}/")

    print('Check if log(RT) differs between choices (W, F)')
    print("mean std log(RT) wait responses:", np.mean(df[df['x11_choice']==0]['x26_logRT']), np.std(df[df['x11_choice']==0]['x26_logRT']))
    print("mean std log(RT) fora responses:", np.mean(df[df['x11_choice']==1]['x26_logRT']), np.std(df[df['x11_choice']==1]['x26_logRT']))
    return df

if __name__ == "__main__":
    processed_df = process_csv_and_export()
