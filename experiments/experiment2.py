import os

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from matplotlib import pyplot as plt

def run_experiment2_interaction(train, test, user_features, output_dir="graphs"):
    """
    For Exp #2, we test user features.
    We'll do the same epoch/learning_rate combos, but pass user_features.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp', learning_rate=lr, random_state=42)
            model.fit(train, user_features=user_features, epochs=e, num_threads=4)

            prec = precision_at_k(model, test, user_features=user_features, k=5).mean()
            au = auc_score(model, test, user_features=user_features).mean()

            results.append({
                'epochs': e,
                'learning_rate': lr,
                'precision': prec,
                'auc': au
            })

    # Plot & Save: Precision@5
    # -------------------------------------------------
    # Create one plot with epochs on X-axis, lines for each learning rate.

    # Group by learning_rate
    lr_dict = {}
    for r in results:
        lr = r['learning_rate']
        if lr not in lr_dict:
            lr_dict[lr] = []
        lr_dict[lr].append(r)

    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        # Sort by epochs so lines go in ascending order
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['precision'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #2 With user features: Precision@5 vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision@5")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment2_precision_interaction.png"))
    plt.close()

    # Plot & Save: AUC
    # -------------------------------------------------
    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['auc'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #2 With user features: AUC vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment2_auc_interaction.png"))
    plt.close()

    return results

def run_experiment2(train_interactions, train_weights, test_interactions, user_features, output_dir="graphs"):
    """
    For Exp #2, we test user features.
    We'll do the same epoch/learning_rate combos, but pass user_features.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp', learning_rate=lr, random_state=42)
            model.fit(train_interactions,sample_weight=train_weights, user_features=user_features, epochs=e, num_threads=4)

            prec = precision_at_k(model, test_interactions, user_features=user_features, k=5).mean()
            au = auc_score(model, test_interactions, user_features=user_features).mean()

            results.append({
                'epochs': e,
                'learning_rate': lr,
                'precision': prec,
                'auc': au
            })

    # Plot & Save: Precision@5
    # -------------------------------------------------
    # Create one plot with epochs on X-axis, lines for each learning rate.

    # Group by learning_rate
    lr_dict = {}
    for r in results:
        lr = r['learning_rate']
        if lr not in lr_dict:
            lr_dict[lr] = []
        lr_dict[lr].append(r)

    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        # Sort by epochs so lines go in ascending order
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['precision'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #2 With user features: Precision@5 vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision@5")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment2_precision_weight.png"))
    plt.close()

    # Plot & Save: AUC
    # -------------------------------------------------
    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['auc'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #2 With user features: AUC vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment2_auc_weight.png"))
    plt.close()

    return results
