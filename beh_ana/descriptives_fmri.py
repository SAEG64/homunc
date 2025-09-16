# %%
import pandas as pd
import glob
import numpy as np

# Find all CSV files starting with "datall"
csv_files = glob.glob("datall*.csv")

total_trials = 0

# Loop through each CSV file
for file in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Count non-null values in x1_id column
        trials_in_file = df['x1_id'].count()
        
        print(f"{file}: {trials_in_file} trials")
        total_trials += trials_in_file
        
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"\nTotal trials across all files: {total_trials}")

# Count participants by trial count
participants_432 = 0
participants_384 = 0

for file in csv_files:
    try:
        df = pd.read_csv(file)
        trials_in_file = df['x1_id'].count()
        
        if trials_in_file == 432:
            participants_432 += 1
        elif trials_in_file == 384:
            participants_384 += 1
            
    except Exception as e:
        continue

print(f"Participants with 432 trials: {participants_432}")
print(f"Participants with 384 trials: {participants_384}")

# Check for unexpected button_pressed values
unexpected_button_values = set()
files_with_unexpected = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Get unique values in button_pressed column
        unique_buttons = df['x9_button_pressed'].dropna().unique()
        # Check if any values are not 2 or 25
        unexpected = [val for val in unique_buttons if val not in [2, 25]]
        if unexpected:
            unexpected_button_values.update(unexpected)
            files_with_unexpected.append(file)
            
    except Exception as e:
        continue

if unexpected_button_values:
    print(f"\nUnexpected button_pressed values found: {sorted(unexpected_button_values)}")
    print(f"Files with unexpected values: {files_with_unexpected}")
else:
    print(f"\nAll participants only have button_pressed values of 2 or 25")
    # Check for RT values >= 2000
    files_with_high_rt = []
    high_rt_count = 0

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Check if any RT values are >= 2000
            high_rt_mask = df['x10_RT'] >= 2000
            if high_rt_mask.any():
                files_with_high_rt.append(file)
                high_rt_count += high_rt_mask.sum()
                
        except Exception as e:
            continue

    if files_with_high_rt:
        print(f"\nParticipants with RT >= 2000ms: {len(files_with_high_rt)}")
        print(f"Total entries with RT >= 2000ms: {high_rt_count}")
        print(f"Files: {files_with_high_rt}")
    else:
        print(f"\nNo participants have RT values >= 2000ms")

    # Calculate filtered trial counts for each participant
    filtered_trial_counts = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Count total trials (non-null x1_id)
            total_trials_file = df['x1_id'].count()
            
            # Count entries with 0 in x6_continuous_energy_trial_start
            zero_energy_count = (df['x6_continuous_energy_trial_start'] == 0).sum()
            
            # Count entries with RT >= 2000
            high_rt_count = (df['x10_RT'] >= 2000).sum()
            
            # Calculate filtered trial count
            filtered_count = total_trials_file - zero_energy_count - high_rt_count
            filtered_trial_counts.append(filtered_count)
        
        except Exception as e:
            continue

    # Calculate mean and standard deviation
    if filtered_trial_counts:
        mean_trials = np.mean(filtered_trial_counts)
        std_trials = np.std(filtered_trial_counts, ddof=1)
        
        print(f"\nFiltered trial counts: {filtered_trial_counts}")
        print(f"Mean filtered trials: {mean_trials:.2f}")
        print(f"Standard deviation: {std_trials:.2f}")

        # Calculate mean and SD of trials with x6_continuous_energy_trial_start == 0
        zero_energy_counts = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                zero_energy_count = (df['x6_continuous_energy_trial_start'] == 0).sum()
                zero_energy_counts.append(zero_energy_count)
            except Exception as e:
                continue
        
        if zero_energy_counts:
            mean_zero_energy = np.mean(zero_energy_counts)
            std_zero_energy = np.std(zero_energy_counts, ddof=1)
            
            print(f"\nTrials with x6_continuous_energy_trial_start == 0:")
            print(f"Mean: {mean_zero_energy:.2f}")
            print(f"Standard deviation: {std_zero_energy:.2f}")

            # Calculate mean and SD of trials with RT >= 2000
            high_rt_counts = []

            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    high_rt_count = (df['x10_RT'] >= 2000).sum()
                    high_rt_counts.append(high_rt_count)
                except Exception as e:
                    continue

            if high_rt_counts:
                mean_high_rt = np.mean(high_rt_counts)
                std_high_rt = np.std(high_rt_counts, ddof=1)
                
                print(f"\nTrials with RT >= 2000ms:")
                print(f"Mean: {mean_high_rt:.2f}")
                print(f"Standard deviation: {std_high_rt:.2f}")

            # Calculate mean and SD of trials with x14_p_foraging_gain in target range
            target_gain_counts = []

            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    target_gain_count = df['x14_p_foraging_gain'].isin([0.4, 0.5, 0.6]).sum()
                    target_gain_counts.append(target_gain_count)
                except Exception as e:
                    continue

            if target_gain_counts:
                mean_target_gain = np.mean(target_gain_counts)
                std_target_gain = np.std(target_gain_counts, ddof=1)
                
                print(f"\nTrials with x14_p_foraging_gain in [0.4, 0.5, 0.6]:")
                print(f"Mean: {mean_target_gain:.2f}")
                print(f"Standard deviation: {std_target_gain:.2f}")