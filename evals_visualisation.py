import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def generate_gov_eval_visuals(results_dir="eval_outputs"):
    # 1. Aggregation Logic
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not all_files:
        print("No CSV results found in directory.")
        return
    
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        # Derive model name from the filename as requested
        model_id = os.path.basename(f).replace("eval_results_", "").replace(".csv", "")
        temp_df['model_filename'] = model_id
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # 2. Metric Engineering
    # Delta K: How many "extra" facts did the model generate? [cite: 313]
    df['delta_k'] = df['afg_k_gen'] - df['afg_k_gt']
    
    # Claim Density: Characters per fact (higher = more word-salad)
    # Average GT length is 499.1 [cite: 184]
    df['char_count'] = df['generated_response'].astype(str).str.len()
    df['claim_density'] = df['char_count'] / df['afg_k_gen'].replace(0, 1) 

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # --- PLOT 1: Score Distribution (Boxplots) ---
    # Shows the range and consistency of F1@K [cite: 282, 312]
    sns.boxplot(ax=axes[0,0], data=df, x='model_filename', y='score')
    axes[0,0].set_title("F1@K Score Distribution per Model Configuration")
    axes[0,0].tick_params(axis='x', rotation=30)

    # --- PLOT 2: The "Word-Salad" Diagnostic ---
    # Correlation between Density (y-axis) and Factuality (x-axis)
    sns.scatterplot(ax=axes[0,1], data=df, x='score', y='claim_density', hue='model_filename', alpha=0.4)
    axes[0,1].set_title("Word-Salad Check: Claim Density vs. Factuality Score")
    axes[0,1].set_ylabel("Characters per Atomic Fact")

    # --- PLOT 3: Delta K (Verbosity) vs Accuracy ---
    # Visualizes the Avg. K-hat - K mentioned in your results table [cite: 311, 313]
    sns.regplot(ax=axes[1,0], data=df, x='delta_k', y='score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    axes[1,0].set_title("Impact of Verbosity (ΔK) on Total Score")
    axes[1,0].set_xlabel("Verbosity (Generated Facts - GT Facts)")

    # --- PLOT 4: Domain Performance Heatmap ---
    # Identifies "Problematic Domains" [cite: 125, 318]
    if 'domain' in df.columns:
        domain_pivot = df.pivot_table(index='domain', columns='model_filename', values='score', aggfunc='mean')
        sns.heatmap(ax=axes[1,1], data=domain_pivot, annot=True, cmap="YlOrRd", cbar_kws={'label': 'Mean F1@K'})
        axes[1,1].set_title("Domain Performance Heatmap (Yellow = Best, Red = Weakest)")
    else:
        axes[1,1].text(0.5, 0.5, "No 'domain' column found in data.\nAdd row['domain'] to worker_verify output.", 
                       ha='center', va='center')

    plt.tight_layout()
    plt.savefig("ogc_eval_visualisation.png", dpi=300)
    print("Dashboard saved as 'ogc_eval_visualisation.png'")

if __name__ == "__main__":
    generate_gov_eval_visuals()