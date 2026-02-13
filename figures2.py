import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def study_neural_network(csv_path):
    # ---------------------------------------------------------
    # 1. DATA LOADING AND PREPROCESSING
    # ---------------------------------------------------------
    print("--- Loading and Cleaning Data ---")
    try:
        # Load data (CSV uses ';' delimiter based on the provided file content)
        df = pd.read_csv(csv_path, sep=';')
        
        # Convert accuracy from string "XX.X%" to float 0.XX
        df['accuracy'] = df['accuracy'].astype(str).str.replace('%', '').astype(float) / 100.0
        
        # Separate Test data from Training/Validation logs
        test_df = df[df['phase'] == 'TEST'].copy()
        history_df = df[df['phase'].isin(['TRAIN', 'VAL'])].copy()
        
        # Ensure we have the requisite diversity in hidden units
        hidden_sizes = sorted(df['hidden_layer_size'].unique())
        print(f"Hidden Layer Sizes found: {hidden_sizes}")
        if len(hidden_sizes) < 5:
            print("Warning: Fewer than 5 different hidden layer sizes detected.")

        seeds = df['random_seed'].unique()
        print(f"Random Seeds found (Initialization robustness): {seeds}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Set visualization style
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # 2. HYPERPARAMETER SENSITIVITY (LR & MOMENTUM)
    # ---------------------------------------------------------
    print("\n--- Analyzing Learning Rate and Momentum ---")
    
    # Aggregate Test Accuracy across all hidden sizes and seeds
    # to see general effectiveness of LR/Momentum pairs
    heatmap_data = test_df.groupby(['learning_rate', 'momentum'])['accuracy'].mean().reset_index()
    heatmap_data = heatmap_data.pivot(index='learning_rate', columns='momentum', values='accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="viridis", cbar_kws={'label': 'Mean Test Accuracy'})
    plt.title('Impact of Learning Rate ($\eta$) and Momentum on Test Accuracy\n(Averaged across hidden sizes and seeds)')
    plt.xlabel('Momentum Coefficient')
    plt.ylabel('Learning Rate ($\eta$)')
    plt.show()

    # ---------------------------------------------------------
    # 3. HIDDEN LAYER SIZE ANALYSIS
    # ---------------------------------------------------------
    print("\n--- Analyzing Hidden Layer Size impact ---")
    
    # Focus on the "best" observed LR/Momentum combo for this architectural plot 
    # to isolate the effect of hidden units.
    # We find the config with the highest max accuracy.
    best_config = test_df.loc[test_df['accuracy'].idxmax()]
    best_lr = best_config['learning_rate']
    best_mom = best_config['momentum']
    
    print(f"Fixing LR={best_lr} and Momentum={best_mom} to study Hidden Layer Sizes...")
    
    architectural_df = test_df[
        (test_df['learning_rate'] == best_lr) & 
        (test_df['momentum'] == best_mom)
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=architectural_df, 
        x='hidden_layer_size', 
        y='accuracy', 
        marker='o',
        errorbar='sd' # Shows the standard deviation across random seeds
    )
    plt.title(f'Test Accuracy vs. Hidden Units\n(LR={best_lr}, Momentum={best_mom}, shaded area = init sensitivity)')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Test Accuracy')
    plt.xscale('log') # Log scale often clearer for doubling sizes (64, 128, 256...)
    plt.xticks(hidden_sizes, hidden_sizes) # Force labels to be the specific sizes
    plt.show()

    # ---------------------------------------------------------
    # 4. TRAINING DYNAMICS & CONVERGENCE
    # ---------------------------------------------------------
    print("\n--- Analyzing Training Dynamics (Convergence) ---")
    
    # We plot Loss curves for different Hidden Sizes to see speed of convergence.
    # We aggregate over seeds to get smooth lines.
    
    subset_history = history_df[
        (history_df['learning_rate'] == best_lr) & 
        (history_df['momentum'] == best_mom) &
        (history_df['phase'] == 'TRAIN') # TODO: Could also plot VAL phase for generalization insights in a separate plot
    ]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=subset_history, 
        x='epoch', 
        y='loss', 
        hue='hidden_layer_size', 
        style='phase', 
        palette='tab10',
        markers=True
    )
    plt.title(f'Learning Curves: Loss over Epochs\n(LR={best_lr}, Momentum={best_mom})')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.legend(title='Hidden Size / Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4.1 VALIDATION DYNAMICS & CONVERGENCE
    # ---------------------------------------------------------

    subset_history = history_df[
        (history_df['learning_rate'] == best_lr) & 
        (history_df['momentum'] == best_mom) &
        (history_df['phase'] == 'VAL')
    ]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=subset_history, 
        x='epoch', 
        y='loss', 
        hue='hidden_layer_size', 
        style='phase', 
        palette='tab10',
        markers=True
    )
    plt.title(f'Learning Curves: Loss over Epochs\n(LR={best_lr}, Momentum={best_mom})')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.legend(title='Hidden Size / Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------
    # 5. STABILITY & SENSITIVITY TO INITIALIZATION
    # ---------------------------------------------------------
    print("\n--- Analyzing Stability (Sensitivity to Initialization) ---")
    
    # Calculate Standard Deviation of Test Accuracy across seeds for every configuration
    stability_df = test_df.groupby(['hidden_layer_size', 'learning_rate', 'momentum'])['accuracy'].agg(['mean', 'std']).reset_index()
    
    # Filter for interesting cases (high mean accuracy) to see if they are stable
    top_performers = stability_df.sort_values(by='mean', ascending=False).head(10)
    
    print("\nTop 10 Configurations by Accuracy (with Stability metric):")
    print(f"{'Hidden Size':<12} | {'LR':<6} | {'Mom':<6} | {'Mean Acc':<10} | {'Std Dev (Stability)':<20}")
    print("-" * 65)
    for _, row in top_performers.iterrows():
        print(f"{int(row['hidden_layer_size']):<12} | {row['learning_rate']:<6} | {row['momentum']:<6} | {row['mean']:.4f}     | {row['std']:.4f}")

    # Visualizing Stability systematic patterns
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stability_df, 
        x='mean', 
        y='std', 
        hue='learning_rate', 
        size='hidden_layer_size',
        sizes=(20, 200),
        palette='deep'
    )
    plt.title('Stability Analysis: Mean Accuracy vs. Initialization Sensitivity (Std Dev)')
    plt.xlabel('Mean Test Accuracy (Higher is better)')
    plt.ylabel('Standard Deviation across Seeds (Lower is better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Hyperparameters')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\n--- Analysis Complete ---")

# Execute the study
if __name__ == "__main__":
    study_neural_network('logs/full.csv')