# %%
import pandas as pd
import numpy as np

# Read the main data_beh.csv file
print("Loading data_beh.csv...")
df_main = pd.read_csv("data_beh.csv")

# Get unique subject IDs
unique_subjects = df_main['x1_id'].unique()
print(f"Found {len(unique_subjects)} unique subjects: {sorted(unique_subjects)}")

# Split data by subjects and analyze each subject like individual files
total_trials = 0
subject_dataframes = {}

# Create individual subject dataframes
for subject_id in unique_subjects:
    subject_df = df_main[df_main['x1_id'] == subject_id].copy()
    subject_dataframes[subject_id] = subject_df
    
    # Count non-null values in x1_id column
    trials_in_subject = subject_df['x1_id'].count()
    
    print(f"Subject {subject_id}: {trials_in_subject} trials")
    total_trials += trials_in_subject

print(f"\nTotal trials across all subjects: {total_trials}")

# Count participants by trial count
participants_432 = 0
participants_384 = 0
trial_counts = []

for subject_id, subject_df in subject_dataframes.items():
    trials_in_subject = subject_df['x1_id'].count()
    trial_counts.append(trials_in_subject)
    
    if trials_in_subject == 432:
        participants_432 += 1
    elif trials_in_subject == 384:
        participants_384 += 1

print(f"Participants with 432 trials: {participants_432}")
print(f"Participants with 384 trials: {participants_384}")
print(f"Other trial counts: {[count for count in trial_counts if count not in [432, 384]]}")

# Check for unexpected button_pressed values
unexpected_button_values = set()
subjects_with_unexpected = []

for subject_id, subject_df in subject_dataframes.items():
    try:
        # Get unique values in button_pressed column
        unique_buttons = subject_df['x9_button_pressed'].dropna().unique()
        # Check if any values are not 2 or 25
        unexpected = [val for val in unique_buttons if val not in [2, 25]]
        if unexpected:
            unexpected_button_values.update(unexpected)
            subjects_with_unexpected.append(subject_id)
            
    except Exception as e:
        continue

if unexpected_button_values:
    print(f"\nUnexpected button_pressed values found: {sorted(unexpected_button_values)}")
    print(f"Subjects with unexpected values: {subjects_with_unexpected}")
else:
    print(f"\nAll participants only have button_pressed values of 2 or 25")
    
    # Check for RT values >= 2000
    subjects_with_high_rt = []
    high_rt_count = 0

    for subject_id, subject_df in subject_dataframes.items():
        try:
            # Check if any RT values are >= 2000
            high_rt_mask = subject_df['x10_RT'] >= 2000
            if high_rt_mask.any():
                subjects_with_high_rt.append(subject_id)
                high_rt_count += high_rt_mask.sum()
                
        except Exception as e:
            continue

    if subjects_with_high_rt:
        print(f"\nParticipants with RT >= 2000ms: {len(subjects_with_high_rt)}")
        print(f"Total entries with RT >= 2000ms: {high_rt_count}")
        print(f"Subjects: {subjects_with_high_rt}")
    else:
        print(f"\nNo participants have RT values >= 2000ms")

    # Calculate filtered trial counts for each participant
    filtered_trial_counts = []

    for subject_id, subject_df in subject_dataframes.items():
        try:
            # Count total trials (non-null x1_id)
            total_trials_subject = subject_df['x1_id'].count()
            
            # Count entries with 0 in x6_continuous_energy_trial_start
            zero_energy_count = (subject_df['x6_continuous_energy_trial_start'] == 0).sum()
            
            # Count entries with RT >= 2000
            high_rt_count = (subject_df['x10_RT'] >= 2000).sum()
            
            # Calculate filtered trial count
            filtered_count = total_trials_subject - zero_energy_count - high_rt_count
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
        
        for subject_id, subject_df in subject_dataframes.items():
            try:
                zero_energy_count = (subject_df['x6_continuous_energy_trial_start'] == 0).sum()
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

            for subject_id, subject_df in subject_dataframes.items():
                try:
                    high_rt_count = (subject_df['x10_RT'] >= 2000).sum()
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

            for subject_id, subject_df in subject_dataframes.items():
                try:
                    target_gain_count = subject_df['x14_p_foraging_gain'].isin([0.4, 0.5, 0.6]).sum()
                    target_gain_counts.append(target_gain_count)
                except Exception as e:
                    continue

            if target_gain_counts:
                mean_target_gain = np.mean(target_gain_counts)
                std_target_gain = np.std(target_gain_counts, ddof=1)
                
                print(f"\nTrials with x14_p_foraging_gain in [0.4, 0.5, 0.6]:")
                print(f"Mean: {mean_target_gain:.2f}")
                print(f"Standard deviation: {std_target_gain:.2f}")

# Additional analysis: Show detailed breakdown per subject
print(f"\n" + "="*60)
print("DETAILED BREAKDOWN PER SUBJECT")
print("="*60)

for subject_id in sorted(unique_subjects):
    subject_df = subject_dataframes[subject_id]
    
    total_trials_subject = subject_df['x1_id'].count()
    zero_energy_count = (subject_df['x6_continuous_energy_trial_start'] == 0).sum()
    high_rt_count = (subject_df['x10_RT'] >= 2000).sum()
    target_gain_count = subject_df['x14_p_foraging_gain'].isin([0.4, 0.5, 0.6]).sum()
    filtered_count = total_trials_subject - zero_energy_count - high_rt_count
    
    print(f"Subject {subject_id}:")
    print(f"  Total trials: {total_trials_subject}")
    print(f"  Zero energy trials: {zero_energy_count}")
    print(f"  High RT trials (>=2000ms): {high_rt_count}")
    print(f"  Target gain trials [0.4,0.5,0.6]: {target_gain_count}")
    print(f"  Filtered trials: {filtered_count}")
    print()