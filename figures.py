import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


FIGURES_DIR = 'figures/'
LOGS_DIR = 'logs/'
CSV_FILES = ["20260211-16-16-17.csv", "20260211-19-53-22.csv"]
CONFIG_COLS = ['hidden_layer_size', 'learning_rate', 'momentum', 'random_seed']


def get_df():
    dfs = [pd.read_csv(os.path.join(LOGS_DIR, f), sep=';') for f in CSV_FILES]
    df = pd.concat(dfs, ignore_index=True)

    # Clean and preprocess
    df['accuracy'] = df['accuracy'].str.replace('%', '').astype(float)
    df['phase'] = df['phase'].str.strip()
    df['hidden_layer_size'] = df['hidden_layer_size'].astype(int)
    df['learning_rate'] = df['learning_rate'].astype(float)
    df['momentum'] = df['momentum'].astype(float)
    df['random_seed'] = df['random_seed'].astype(int)
    return df


# 1. Plot accuracy/loss curves for each config (train/val)
def plot_curves(df):
    configs = df[CONFIG_COLS[:-1]].drop_duplicates()
    for _, cfg in configs.iterrows():
        mask = (df['hidden_layer_size'] == cfg['hidden_layer_size']) & \
               (df['learning_rate'] == cfg['learning_rate']) & \
               (df['momentum'] == cfg['momentum'])
        for seed in df[mask]['random_seed'].unique():
            sub = df[mask & (df['random_seed'] == seed)]
            plt.figure(figsize=(10,4))
            for phase in ['TRAIN', 'VAL']:
                phase_df = sub[sub['phase'] == phase]
                plt.plot(phase_df['epoch'], phase_df['accuracy'], label=f'{phase} acc')
            plt.title(f"Acc: HL={cfg['hidden_layer_size']} η={cfg['learning_rate']} μ={cfg['momentum']} seed={seed}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.tight_layout()

            os.makedirs(os.path.join(FIGURES_DIR, f"HL{cfg['hidden_layer_size']}"), exist_ok=True)
            os.makedirs(os.path.join(FIGURES_DIR, f"HL{cfg['hidden_layer_size']}", f"LR{cfg['learning_rate']}"), exist_ok=True)
            os.makedirs(os.path.join(FIGURES_DIR, f"HL{cfg['hidden_layer_size']}", f"LR{cfg['learning_rate']}", f"M{cfg['momentum']}"), exist_ok=True)
            fname = os.path.join(FIGURES_DIR, f"HL{cfg['hidden_layer_size']}", f"LR{cfg['learning_rate']}", f"M{cfg['momentum']}", f"SEED{seed}.png")
            plt.savefig(fname)
            plt.close()


# 2. Test accuracy for each config (boxplot for seeds)
def plot_test_accuracy(df):
    test_df = df[df['phase'] == 'TEST']
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=test_df,
        x='hidden_layer_size',
        y='accuracy',
        hue='learning_rate',
        palette='Set2'
    )
    plt.title("Test Accuracy by Hidden Layer Size and Learning Rate")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Hidden Layer Size")
    plt.legend(title="Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "test_accuracy_boxplot.png"))
    plt.close()


# 3. Convergence speed (epochs to best val acc)
def plot_convergence(df):
    val_df = df[df['phase'] == 'VAL']
    best_epochs = val_df.groupby(CONFIG_COLS).apply(lambda x: x.loc[x['accuracy'].idxmax(), 'epoch'])
    best_epochs = best_epochs.reset_index(name='best_epoch')
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=best_epochs,
        x='hidden_layer_size',
        y='best_epoch',
        hue='learning_rate',
        palette='Set1'
    )
    plt.title("Epochs to Best Validation Accuracy")
    plt.ylabel("Epoch")
    plt.xlabel("Hidden Layer Size")
    plt.legend(title="Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "convergence_boxplot.png"))
    plt.close()


# 4. Stability: std of test accuracy across seeds
def plot_stability(df):
    test_df = df[df['phase'] == 'TEST']
    grouped = test_df.groupby(['hidden_layer_size', 'learning_rate', 'momentum'])['accuracy'].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(12,6))
    for lr in grouped['learning_rate'].unique():
        sub = grouped[grouped['learning_rate'] == lr]
        plt.errorbar(sub['hidden_layer_size'], sub['mean'], yerr=sub['std'], label=f'LR={lr}', capsize=4, marker='o')
    plt.title("Stability of Test Accuracy (Std across Seeds)")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stability_errorbar.png"))
    plt.close()


# Run all plots
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = get_df()
    plot_curves(df)
    plot_test_accuracy(df)
    plot_convergence(df)
    plot_stability(df)