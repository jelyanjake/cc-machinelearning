import pandas as pd
import nlpaug.augmenter.word as naw

# --- CONFIGURATION ---
input_file = 'complaints.csv'
output_file = 'complaints_augmented.csv'
samples_per_row = 2  # How many new variations to create per original row

# --- 1. SETUP AUGMENTER ---
# Using WordNet for synonym replacement
aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3) 

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Make sure the file is in the same folder.")
    exit()

# --- 3. AUGMENTATION LOOP ---
new_rows = []

print("Augmenting data... this may take a moment.")

for index, row in df.iterrows():
    original_text = row['text']
    label = row['label']
    
    # Generate 'n' variations
    # aug.augment returns a list of strings when n > 1
    augmented_texts = aug.augment(original_text, n=samples_per_row)
    
    # Ensure it's a list (sometimes nlpaug returns a single string if n=1)
    if isinstance(augmented_texts, str):
        augmented_texts = [augmented_texts]
        
    # Add each new variation to our list with the SAME label as the original
    for new_text in augmented_texts:
        new_rows.append({
            'text': new_text,
            'label': label
        })

# --- 4. COMBINE AND SAVE ---
# Create a DataFrame from the new rows
new_df = pd.DataFrame(new_rows)

# Combine original data with new data
final_df = pd.concat([df, new_df], ignore_index=True)

# Shuffle the dataset (Good practice for Machine Learning)
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
final_df.to_csv(output_file, index=False)

print("-" * 30)
print(f"Success! Created {len(new_rows)} new examples.")
print(f"Total dataset size: {len(final_df)} rows.")
print(f"Saved to: {output_file}")