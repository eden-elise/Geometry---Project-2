import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data from dataset.json
try:
    with open('dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: dataset.json not found. Please ensure the file is in the same directory.")
    data = []

if data:
    # 2. Convert to DataFrame and Filter
    df_raw = pd.DataFrame(data)

    # Ensure we only look at the target languages
    target_langs = ['English', 'Chinese', 'Arabic']
    df = df_raw[df_raw['language'].isin(target_langs)].copy()

    # 3. Feature Engineering
    df['hull_point_count'] = df['hull_points'].apply(len)
    df['points_per_char'] = df['hull_point_count'] / df['num_chars']

    # 4. Calculate Stats (Mean and Variance)
    stats = df.groupby('language')[['hull_point_count', 'points_per_char']].agg(['mean', 'var'])
    
    # Calculate Standard Deviation for the "Range" visualization
    # Range = Mean +/- StdDev
    stats[('hull_point_count', 'std')] = np.sqrt(stats[('hull_point_count', 'var')])
    stats[('points_per_char', 'std')] = np.sqrt(stats[('points_per_char', 'var')])

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    languages = stats.index
    x = np.arange(len(languages))

    # Chart 1: Total Hull Points
    ax1.bar(languages, 
            stats['hull_point_count']['mean'], 
            yerr=stats['hull_point_count']['std'], 
            capsize=10, 
            color='#3498db', 
            alpha=0.8, 
            label='Mean ± Std Dev')
    
    ax1.set_title('Mean # of Hull Points (with Std Dev Range)', fontweight='bold')
    ax1.set_ylabel('Number of Points')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend()

    # Chart 2: Points per Character
    ax2.bar(languages, 
            stats['points_per_char']['mean'], 
            yerr=stats['points_per_char']['std'], 
            capsize=10, 
            color='#2ecc71', 
            alpha=0.8, 
            label='Mean ± Std Dev')
    
    ax2.set_title('Mean # of Hull Points Per Character (with Std Dev Range)', fontweight='bold')
    ax2.set_ylabel('Ratio (Points/Char)')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print summary for verification
    print(stats[['hull_point_count', 'points_per_char']])