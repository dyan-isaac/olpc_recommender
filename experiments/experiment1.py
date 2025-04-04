import os

import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def run_experiment1_interaction(train, test, mode, output_dir="graphs"):
    """
    For Exp #1, we do NOT use user or item features.
    We test combos of (epochs, learning_rate).
    Returns a list of results dicts.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp', learning_rate=lr, random_state=42)
            model.fit(train, epochs=e, num_threads=4)

            train_prec = precision_at_k(model, train, k=5).mean()
            train_au = auc_score(model, train).mean()

            test_prec = precision_at_k(model, test, k=5).mean()
            test_au = auc_score(model, test).mean()

            results.append({
                'epochs': e,
                'learning_rate': lr,
                'test_prec': test_prec,
                'test_auc': test_au,
                'train_prec': train_prec,
                'train_auc': train_au
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
            y = [v['test_prec'] for v in vals_sorted]
            plt.plot(x, y, marker='o', label=f"LR={lr}")

        plt.title("Experiment #1 No user/item features: Precision@5 vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Precision@5")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'experiment1_precision_{mode}.png'))
        plt.close()

        # Plot & Save: AUC
        # -------------------------------------------------
        plt.figure(figsize=(6, 4))
        for lr, vals in lr_dict.items():
            vals_sorted = sorted(vals, key=lambda x: x['epochs'])
            x = [v['epochs'] for v in vals_sorted]
            y = [v['test_auc'] for v in vals_sorted]
            plt.plot(x, y, marker='o', label=f"LR={lr}")

        plt.title("Experiment #1 No user/item features: AUC vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'experiment1_auc_{mode}.png'))
        plt.close()

    return results


def run_experiment1(train_interactions, train_weights, test_interactions, mode, output_dir="graphs"):
    """
    For Exp #1, we do NOT use user or item features.
    We test combos of (epochs, learning_rate).
    Returns a list of results dicts.
    """
    epoch_values = [10, 20, 30, 50]
    learning_rates = [0.005, 0.01, 0.02, 0.05]

    results = []

    for e in epoch_values:
        for lr in learning_rates:
            model = LightFM(loss='warp',learning_rate=lr, learning_schedule='adadelta', random_state=42, user_alpha=0.01, item_alpha=0.01)
            model.fit(train_interactions, sample_weight=train_weights, epochs=e, num_threads=4)

            train_prec = precision_at_k(model, train_interactions, k=5).mean()
            train_au = auc_score(model, train_interactions).mean()

            test_prec = precision_at_k(model, test_interactions, k=5).mean()
            test_au = auc_score(model, test_interactions).mean()

            results.append({
                'epochs': e,
                'learning_rate': lr,
                'test_prec': test_prec,
                'test_auc': test_au,
                'train_prec': train_prec,
                'train_auc': train_au
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
            y = [v['test_prec'] for v in vals_sorted]
            plt.plot(x, y, marker='o', label=f"LR={lr}")

        plt.title("Experiment #1 No user/item features: Precision@5 vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Precision@5")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'experiment1_precision_{mode}.png'))
        plt.close()

        # Plot & Save: AUC
        # -------------------------------------------------
        plt.figure(figsize=(6, 4))
        for lr, vals in lr_dict.items():
            vals_sorted = sorted(vals, key=lambda x: x['epochs'])
            x = [v['epochs'] for v in vals_sorted]
            y = [v['test_auc'] for v in vals_sorted]
            plt.plot(x, y, marker='o', label=f"LR={lr}")

        plt.title("Experiment #1 No user/item features: AUC vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'experiment1_auc_{mode}.png'))
        plt.close()

    return results
