# %%
import pandas as pd

# Import the CSV file
df = pd.read_csv('data_fmri.csv')

# Calculate average number of trials per subject
avg_trials_per_subject = df.groupby('x1_id').size().mean()

print(f"Average number of trials per subject: {avg_trials_per_subject}")