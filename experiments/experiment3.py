import os

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from matplotlib import pyplot as plt

def run_experiment3_interaction(train, test, user_features=None, item_features=None, output_dir="graphs"):
    """
    For Exp #3, we pass both user and item features if available.
    Tuning epochs and learning_rate as well.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp', learning_rate=lr, random_state=42)
            model.fit(train,
                      user_features=user_features,
                      item_features=item_features,
                      epochs=e,
                      num_threads=4)

            prec = precision_at_k(model, test, user_features=user_features, item_features=item_features, k=5).mean()
            au = auc_score(model, test, user_features=user_features, item_features=item_features).mean()

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

    plt.title("Experiment #3 With user and item features: Precision@5 vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision@5")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment3_precision_interaction.png"))
    plt.close()

    # Plot & Save: AUC
    # -------------------------------------------------
    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['auc'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #3 With user and item features: AUC vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment3_auc_interaction.png"))
    plt.close()

    return results


def run_experiment3(train_interactions, train_weights, test_interactions, user_features=None, item_features=None, output_dir="graphs"):
    """
    For Exp #3, we pass both user and item features if available.
    Tuning epochs and learning_rate as well.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp', learning_rate=lr, random_state=42)
            model.fit(train_interactions,
                      sample_weight=train_weights,
                      user_features=user_features,
                      item_features=item_features,
                      epochs=e,
                      num_threads=4)

            prec = precision_at_k(model, test_interactions, user_features=user_features, item_features=item_features, k=5).mean()
            au = auc_score(model, test_interactions, user_features=user_features, item_features=item_features).mean()

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

    plt.title("Experiment #3 With user and item features: Precision@5 vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision@5")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment3_precision_weight.png"))
    plt.close()

    # Plot & Save: AUC
    # -------------------------------------------------
    plt.figure(figsize=(6, 4))
    for lr, vals in lr_dict.items():
        vals_sorted = sorted(vals, key=lambda x: x['epochs'])
        x = [v['epochs'] for v in vals_sorted]
        y = [v['auc'] for v in vals_sorted]
        plt.plot(x, y, marker='o', label=f"LR={lr}")

    plt.title("Experiment #3 With user and item features: AUC vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, "experiment3_auc_weight.png"))
    plt.close()

    return results
