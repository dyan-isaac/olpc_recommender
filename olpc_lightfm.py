import json
import datetime
import pickle
import os
import pandas as pd
import numpy as np
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

class OLPCLightFM:
    """
    Class for preprocessing OLPC school-app usage data and NAPLAN scores.
    Handles data loading, normalization, and preparation for LightFM experiments.
    """

    def __init__(self, data_dir="data", kaggle_dir="/kaggle/input/olpc-log/", use_kaggle=False):
        """
        Initialize the preprocessor with data paths.

        Parameters:
        -----------
        data_dir : str, default="data"
            Local directory containing the data files
        kaggle_dir : str, default="/kaggle/input/olpc-log/"
            Kaggle dataset directory (used only if use_kaggle=True)
        use_kaggle : bool, default=False
            Whether to use Kaggle paths or local paths
        """
        self.data_dir = data_dir
        self.kaggle_dir = kaggle_dir
        self.use_kaggle = use_kaggle

        # Dataset attributes to be populated
        self.naplan_df = None
        self.raw_usage_df = None
        self.normalized_usage_df = None
        self.merged_df = None

        # LightFM dataset and matrices
        self.dataset = None
        self.interactions = None
        self.weights = None
        self.user_features = None
        self.item_features = None

    def get_data_path(self):
        """Get the path to OLPC dataset files."""
        if self.use_kaggle:
            return self.kaggle_dir
        else:
            return os.path.join(os.getcwd(), self.data_dir)

    def load_naplan_scores(self):
        """
        Load NAPLAN scores and apply preprocessing.

        Returns:
        --------
        pd.DataFrame
            Preprocessed NAPLAN scores dataframe
        """
        print("1. Loading naplan_scores.csv...")
        performance_path = os.path.join(self.get_data_path(), "naplan_minmax.csv")
        self.naplan_df = pd.read_csv(performance_path)

        # Binning avg_score per school
        bins = [0, 400, 420, 440, 460, float('inf')]
        labels = ['bin1','bin2','bin3','bin4','bin5']

        self.naplan_df['score_bin'] = pd.cut(self.naplan_df['avg_score'], bins=bins, labels=labels, right=False)

        print("1.1 Applying minmax normalisation for NAPLAN average score")
        scaler = StandardScaler()
        self.naplan_df['norm_avg_score'] = scaler.fit_transform(self.naplan_df[['avg_score']])

        self.naplan_df['naplan_score_scaled'] = self.naplan_df['minmax_norm'].round(2)

        return self.naplan_df

    def load_app_usage(self):
        """
        Load app usage data and perform initial cleaning.

        Returns:
        --------
        pd.DataFrame
            Initial cleaned app usage dataframe
        """
        print("\n2. Loading aggregated app usage for school-device-app...")
        usage_path = os.path.join(self.get_data_path(), "aggregated-school-device-app-category-rating-duration.csv")
        usage_df = pd.read_csv(usage_path)
        usage_df.rename(columns={
            'school id': 'school_id',
            'app id': 'app_id',
            'total duration': 'duration',
            'device id': 'device_id',
            'category id': 'category_id',
            'app rating': 'app_rating'
        }, inplace=True)

        self.raw_usage_df = usage_df[['school_id', 'device_id', 'app_id', 'category_id', 'app_rating', 'duration']]
        return self.raw_usage_df

    def normalize_app_usage(self):
        """
        Normalize app usage data by device count and apply transformations.

        Returns:
        --------
        pd.DataFrame
            Normalized app usage dataframe
        """
        if self.raw_usage_df is None:
            self.load_app_usage()

        print("\n3. Normalizing aggregated app usage duration per school-app pair...")
        usage_df = self.raw_usage_df

        # Calculate device count per school
        devices_per_school = usage_df.groupby('school_id')['device_id'].nunique().reset_index().rename(
            columns={'device_id': 'device_count'})
        usage_df = usage_df.merge(devices_per_school, on='school_id', how='left')

        # Aggregate by school-app-category-rating
        usage_agg = usage_df.groupby(['school_id', 'app_id', 'category_id', 'app_rating', 'device_count']).agg(
            {'duration': 'sum'}).reset_index()

        print("3.1 Removing app duration that yield to 0 app usage duration.")
        # Remove zero durations
        count_zero_durations = (usage_agg['duration'] == 0).sum()
        print(f"    Number of rows with zero duration to be removed: {count_zero_durations}")
        usage_agg = usage_agg[usage_agg['duration'] > 0]

        print("3.2 Penalising duration with number of devices per school app usage.")
        usage_agg['duration_device_ratio'] = usage_agg['duration'] / usage_agg['device_count']

        print("3.3 Performing log transformation for duration_device_ratio")
        usage_agg['norm_duration_device_ratio'] = np.log1p(usage_agg['duration_device_ratio'])

        print("3.4 After log transform, applying MinMax scaling to normalize the data between 0 and 1.")
        scaler = MinMaxScaler()
        usage_agg['scaled_duration'] = scaler.fit_transform(usage_agg[['norm_duration_device_ratio']])
        usage_agg['scaled_duration'] = usage_agg['scaled_duration'].round(2)

        self.normalized_usage_df = usage_agg[['school_id', 'app_id', 'category_id', 'app_rating', 'scaled_duration']]
        return self.normalized_usage_df

    def merge_datasets(self):
        """
        Merge normalized app usage with NAPLAN scores.

        Returns:
        --------
        pd.DataFrame
            Merged dataframe with app usage and NAPLAN scores
        """
        if self.naplan_df is None:
            self.load_naplan_scores()

        if self.normalized_usage_df is None:
            self.normalize_app_usage()

        print("\n4. Merging app usage data with NAPLAN scores...")
        self.merged_df = self.normalized_usage_df.merge(
            self.naplan_df[['school_id', 'naplan_score_scaled']],
            on='school_id',
            how='left'
        )

        return self.merged_df

    def prepare_lightfm_dataset(self):
        """
        Prepare dataset for LightFM, creating interactions and feature matrices for all experiment types:
        1. No weights (binary interactions)
        2. Duration as weights
        3. NAPLAN scores as weights

        Returns:
        --------
        dict
            Dictionary containing all interaction matrices and feature matrices for experiments
        """
        if self.merged_df is None:
            self.merge_datasets()

        print("\n5. Preparing data for LightFM...")
        # Create LightFM dataset
        self.dataset = Dataset()

        school_ids = self.merged_df['school_id'].astype('category')
        app_ids = self.merged_df['app_id'].astype('category')
        category_ids = self.merged_df['category_id'].astype(int)

        unique_scores_sorted = sorted(self.merged_df['naplan_score_scaled'].unique())
        user_feature_labels = [f"naplan_score:{b}" for b in unique_scores_sorted]

        unique_cats = category_ids.unique().tolist()  # now ints
        unique_ratings = self.merged_df['app_rating'].unique().tolist()

        cat_feature_labels = [f"category_id:{cat}" for cat in unique_cats]
        rating_feature_labels = [f"app_rating:{rating}" for rating in unique_ratings]

        item_feature_labels = cat_feature_labels + rating_feature_labels

        # Fit dataset with users (schools), items (apps), and their features
        self.dataset.fit(
            users=school_ids.unique().tolist(),
            items=app_ids.unique().tolist(),
            user_features=user_feature_labels,
            item_features=item_feature_labels
        )

        # Build interaction matrices for all three experiment types

        # 1. Binary interactions (no weights)
        print("5.1 Building binary interaction matrix (Experiment 1: No Weights)...")
        self.interactions, _ = self.dataset.build_interactions(list(zip(
            self.merged_df['school_id'],
            self.merged_df['app_id']
        )))

        # 2. Duration-weighted interactions
        print("5.2 Building duration-weighted interaction matrix (Experiment 2: Duration Weights)...")
        self.duration_interactions, self.duration_weights = self.dataset.build_interactions(list(zip(
            self.merged_df['school_id'],
            self.merged_df['app_id'],
            self.merged_df['scaled_duration']
        )))

        # 3. NAPLAN-weighted interactions
        print("5.3 Building NAPLAN-weighted interaction matrix (Experiment 3: NAPLAN Weights)...")
        self.naplan_interactions, self.naplan_weights = self.dataset.build_interactions(list(zip(
            self.merged_df['school_id'],
            self.merged_df['app_id'],
            self.merged_df['naplan_score_scaled']
        )))

        # Build user features (NAPLAN scores)
        print("5.4 Building user features matrix...")
        user_features_df = (
                self.merged_df[['school_id', 'naplan_score_scaled']]
                .drop_duplicates(subset='school_id')
                .dropna(subset=['naplan_score_scaled'])  # ensure no missing bins
            )
        self.user_features = self.dataset.build_user_features(
            (
                (row.school_id, [f"naplan_score:{row.naplan_score_scaled}"])
                for _, row in user_features_df.iterrows()
            )
        )

        # Build item features (category and rating)
        print("5.5 Building item features matrices...")
        item_features_df = (
            self.merged_df[['app_id', 'category_id', 'app_rating']]
            .drop_duplicates(subset='app_id')
            .dropna(subset=['category_id'])
            .dropna(subset=['app_rating'])
        )

        self.item_feature_configs = {}
        for config in [['category_id'], ['category_id', 'app_rating']]:
            feature_cols = ['app_id'] + config
            temp_df = item_features_df[feature_cols].copy()
            if 'category_id' in config:
                temp_df['category_id'] = temp_df['category_id'].astype(int)
            if 'app_rating' in config:
                temp_df['app_rating'] = temp_df['app_rating'].round(1)
            feature_matrix = self.dataset.build_item_features(
                (
                    (
                        row.app_id,
                        [
                            f"{col}:{int(row[col])}" if col == 'category_id' else f"{col}:{row[col]}"
                            for col in config
                        ]
                    )
                    for _, row in temp_df.iterrows()
                )
            )
            key = '_'.join(config)
            self.item_feature_configs[key] = feature_matrix

        # Organize all interaction matrices in a dictionary
        self.interaction_matrices = {
            "no_weights": self.interactions,
            "duration_weights": self.duration_interactions,
            "naplan_weights": self.naplan_interactions
        }

        self.weights_matrices = {
            "no_weights": None,
            "duration_weights": self.duration_weights,
            "naplan_weights": self.naplan_weights
        }

        return {
            "interaction_matrices": self.interaction_matrices,
            "user_features": self.user_features,
            "item_features": self.item_feature_configs
        }

    def preprocess_data(self):
        """
        Run the full preprocessing pipeline.

        Returns:
        --------
        dict
            Dictionary with all preprocessed data and matrices
        """
        self.load_naplan_scores()
        self.load_app_usage()
        self.normalize_app_usage()
        self.merge_datasets()
        self.prepare_lightfm_dataset()

        return {
            "interaction_matrices": self.interaction_matrices,  # Contains all three types of matrices
            "weights_matrices": self.weights_matrices,
            "user_features": self.user_features,
            "item_features": self.item_feature_configs,
            "dataset": self.dataset,
            "merged_df": self.merged_df
        }

    def run_experiments(self, standard_params=None):
        """
        Run all experiments with different weight types and feature combinations:

        Experiment 1: No weights (binary interactions)
        Experiment 2: Duration as weights
        Experiment 3: NAPLAN as weights

        Each experiment  1 and 2 has 4 feature combinations:
        - No features
        - NAPLAN as user feature
        - NAPLAN + app category as features
        - NAPLAN + app category + rating as features

        Parameters:
        -----------
        standard_params : dict, optional
            Standard hyperparameters to use across all experiments

        Returns:
        --------
        dict
            Dictionary with experiment results and summary DataFrame
        """
        if not hasattr(self, 'interaction_matrices'):
            self.preprocess_data()

        # Import required functions
        from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
        from lightfm import LightFM
        from lightfm.cross_validation import random_train_test_split

        # Define feature combinations
        feature_combinations = {
            "1_no_features": {
                "user_features": None,
                "item_features": None,
                "name": "No Features"
            },
            "2_user_naplan": {
                "user_features": self.user_features,
                "item_features": None,
                "name": "NAPLAN User Features"
            },
            "3_user_naplan_item_category": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id"],
                "name": "NAPLAN + App Category"
            },
            "4_user_naplan_item_all": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "NAPLAN + App Category + Rating"
            },
            "5_no_user_feature_item_category": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id"],
                "name": "No User Feature + App Category"
            },
            "6_no_user_feature_item_all": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "No User Feature + App Category + Rating"
            }
        }

        # Define weight types
        weight_types = {
            "1_no_weights": {
                "interactions": self.interaction_matrices["no_weights"],
                "weights": None,
                "name": "No Weights"
            },
            "2_duration_weights": {
                "interactions": self.interaction_matrices["duration_weights"],
                "weights": self.weights_matrices["duration_weights"], # Duration weights
                "name": "Duration Weights"
            },
            "3_naplan_weights": {
                "interactions": self.interaction_matrices["naplan_weights"],
                "weights": self.weights_matrices["naplan_weights"], # NAPLAN weights
                "name": "NAPLAN Weights"
            }
        }

        # Set default parameters if not provided
        if standard_params is None:
            standard_params = {'epochs': 50, 'learning_rate': 0.004}

        print(f"Using standard parameters for all experiments: {standard_params}")

        # Run all experiment combinations
        results = {}
        experiment_data = []

        for weight_key, weight_config in weight_types.items():
            print(f"\nRunning experiments with {weight_config['name']}...")

            for feature_key, feature_config in feature_combinations.items():
                experiment_id = f"Exp {weight_key}_{feature_key}"
                experiment_name = f"{weight_config['name']}, {feature_config['name']}"

                print(f"\nRunning {experiment_id}: {experiment_name}")

                # Split data into train and test
                train_interactions, test_interactions = random_train_test_split(
                    weight_config['interactions'],
                    test_percentage=0.2,
                    random_state=np.random.RandomState(3)
                )

                # Also split weights if they exist
                train_weights = None
                if weight_config['weights'] is not None:
                    train_weights, _ = random_train_test_split(
                        weight_config['weights'],
                        test_percentage=0.2,
                        random_state=np.random.RandomState(3)
                    )

                # Train model with specified parameters
                model = LightFM(
                    loss="warp",
                    learning_schedule="adadelta",
                    learning_rate=standard_params['learning_rate'],
                    random_state=3
                )

                model.fit(
                    train_interactions,
                    sample_weight=train_weights,  # Correctly pass weights here
                    user_features=feature_config['user_features'],
                    item_features=feature_config['item_features'],
                    epochs=standard_params['epochs'],
                    verbose=True
                )

                # Evaluate model
                train_p = precision_at_k(model, test_interactions=train_interactions, k=5, user_features=feature_config['user_features'],
                                      item_features=feature_config['item_features']).mean()
                train_r = recall_at_k(model, test_interactions=train_interactions, k=5, user_features=feature_config['user_features'],
                                    item_features=feature_config['item_features']).mean()
                train_a = auc_score(model, test_interactions=train_interactions, user_features=feature_config['user_features'],
                                  item_features=feature_config['item_features']).mean()

                test_p = precision_at_k(model, test_interactions=test_interactions, k=5, user_features=feature_config['user_features'],
                                     item_features=feature_config['item_features']).mean()
                test_r = recall_at_k(model, test_interactions=test_interactions, k=5, user_features=feature_config['user_features'],
                                   item_features=feature_config['item_features']).mean()
                test_a = auc_score(model, test_interactions=test_interactions, user_features=feature_config['user_features'],
                                 item_features=feature_config['item_features']).mean()

                # Store results
                metrics = {
                    'train_precision': train_p,
                    'test_precision': test_p,
                    'train_recall': train_r,
                    'test_recall': test_r,
                    'train_auc': train_a,
                    'test_auc': test_a
                }

                results[experiment_id] = {
                    'model': model,
                    'metrics': metrics,
                    'params': standard_params,
                    'name': experiment_name
                }

                # Add to summary data
                experiment_data.append({
                    'Experiment ID': experiment_id,
                    'Experiment': experiment_name,
                    'Weight Type': weight_config['name'],
                    'Features': feature_config['name'],
                    'Precision@5': test_p,
                    'Recall@5': test_r,
                    'AUC': test_a
                })

                print(f"Results for {experiment_id}:")
                print(f"  Precision@5: {test_p:.4f}")
                print(f"  Recall@5: {test_r:.4f}")
                print(f"  AUC: {test_a:.4f}")

        # Create summary DataFrame
        results_df = pd.DataFrame(experiment_data)

        # Save results
        results_df.to_csv('all_experiment_results_summary.csv', index=False)

        print("\nSummary of all experiments:")
        print(results_df)

        return {
            'results': results,
            'summary_df': results_df
        }

    def tune_hyperparameters(self, weight_key="1_no_weights", feature_keys=None,
                        epochs_list=None, lr_list=None, output_dir="graphs"):
        """
        Run hyperparameter tuning experiments to find the best epoch and learning rate values
        for multiple feature configurations.

        Parameters:
        -----------
        weight_key : str, default="1_no_weights"
            Which weight configuration to use (1_no_weights, 2_duration_weights, 3_naplan_weights)
        feature_keys : list, optional
            List of feature configurations to test. If None, uses ["4_user_naplan_item_all"]
        epochs_list : list, optional
            List of epoch values to test. Defaults to [10, 20, 30, 50, 70]
        lr_list : list, optional
            List of learning rate values to test. Defaults to [0.001, 0.002, 0.005, 0.01, 0.02]
        output_dir : str, default="graphs"
            Directory to save visualization graphs

        Returns:
        --------
        dict
            Dictionary with tuning results and best parameters for each feature configuration
        """
        import matplotlib.pyplot as plt
        from lightfm import LightFM
        from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Set default feature keys if not provided
        if feature_keys is None:
            feature_keys = ["4_user_naplan_item_all"]
        elif isinstance(feature_keys, str):
            feature_keys = [feature_keys]  # Convert single string to list

        # Default hyperparameter values to test
        if epochs_list is None:
            epochs_list = [10, 20, 30, 50, 70]
        if lr_list is None:
            lr_list = [0.001, 0.002, 0.005, 0.01, 0.02]


        # Define feature combinations
        feature_combinations = {
            "1_no_features": {
                "user_features": None,
                "item_features": None,
                "name": "No Features"
            },
            "2_user_naplan": {
                "user_features": self.user_features,
                "item_features": None,
                "name": "NAPLAN User Features"
            },
            "3_user_naplan_item_category": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id"],
                "name": "NAPLAN + App Category"
            },
            "4_user_naplan_item_all": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "NAPLAN + App Category + Rating"
            },
            "5_no_user_feature_item_category": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id"],
                "name": "No User Feature + App Category"
            },
            "6_no_user_feature_item_all": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "No User Feature + App Category + Rating"
            }
        }

        # Define weight types
        weight_types = {
            "1_no_weights": {
                "interactions": self.interaction_matrices["no_weights"],
                "weights": None,
                "name": "No Weights"
            },
            "2_duration_weights": {
                "interactions": self.interaction_matrices["duration_weights"],
                "weights": self.weights_matrices["duration_weights"], # Duration weights
                "name": "Duration Weights"
            },
            "3_naplan_weights": {
                "interactions": self.interaction_matrices["naplan_weights"],
                "weights": self.weights_matrices["naplan_weights"], # NAPLAN weights
                "name": "NAPLAN Weights"
            }
        }

        # Get the weight configuration
        weight_config = weight_types[weight_key]

        # Dictionary to store results for all feature configurations
        all_results = {}
        best_params_overall = {}

        # Split data into train and test (shared across all feature configs)
        train_interactions, test_interactions = random_train_test_split(
            weight_config['interactions'],
            test_percentage=0.2,
            random_state=np.random.RandomState(3)
        )

        # Split weights if they exist
        train_weights = None
        if weight_config['weights'] is not None:
            train_weights, _ = random_train_test_split(
                weight_config['weights'],
                test_percentage=0.2,
                random_state=np.random.RandomState(3)
            )

        # Iterate through each feature configuration
        for feature_key in feature_keys:
            feature_config = feature_combinations[feature_key]
            experiment_name = f"{weight_config['name']}, {feature_config['name']}"

            print(f"\nRunning hyperparameter tuning for: {experiment_name}")

            # Run experiments with different hyperparameters
            results = []

            for epochs in epochs_list:
                for lr in lr_list:
                    print(f"Testing: epochs={epochs}, learning_rate={lr}")

                    # Train model
                    model = LightFM(
                        loss="warp",
                        learning_schedule="adadelta",
                        learning_rate=lr,
                        random_state=42,
                        user_alpha=0.01, item_alpha=0.01
                    )

                    model.fit(
                        train_interactions,
                        sample_weight=train_weights,
                        user_features=feature_config['user_features'],
                        item_features=feature_config['item_features'],
                        epochs=epochs,
                        num_threads=4
                    )

                    # Evaluate model
                    train_p = precision_at_k(model, test_interactions=train_interactions, k=5,
                                          user_features=feature_config['user_features'],
                                          item_features=feature_config['item_features']).mean()
                    train_r = recall_at_k(model, test_interactions=train_interactions, k=5,
                                        user_features=feature_config['user_features'],
                                        item_features=feature_config['item_features']).mean()
                    train_a = auc_score(model, test_interactions=train_interactions,
                                      user_features=feature_config['user_features'],
                                      item_features=feature_config['item_features']).mean()

                    test_p = precision_at_k(model, test_interactions=test_interactions, k=5,
                                         user_features=feature_config['user_features'],
                                         item_features=feature_config['item_features']).mean()
                    test_r = recall_at_k(model, test_interactions=test_interactions, k=5,
                                       user_features=feature_config['user_features'],
                                       item_features=feature_config['item_features']).mean()
                    test_a = auc_score(model, test_interactions=test_interactions,
                                     user_features=feature_config['user_features'],
                                     item_features=feature_config['item_features']).mean()

                    # Store results
                    result = {
                        'epochs': epochs,
                        'learning_rate': lr,
                        'train_precision': train_p,
                        'test_precision': test_p,
                        'train_recall': train_r,
                        'test_recall': test_r,
                        'train_auc': train_a,
                        'test_auc': test_a,
                        'feature_config': feature_config['name']
                    }
                    results.append(result)

            # Create DataFrame with results
            results_df = pd.DataFrame(results)

            # Find best configuration based on test AUC
            best_result = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"\nBest configuration for {feature_config['name']}: epochs={best_result['epochs']}, learning_rate={best_result['learning_rate']}")
            print(f"Test AUC: {best_result['test_auc']:.4f}, Test Precision: {best_result['test_precision']:.4f}")

            # Create visualizations
            # Plot Precision@5
            plt.figure(figsize=(10, 6))
            for lr in sorted(results_df['learning_rate'].unique()):
                lr_data = results_df[results_df['learning_rate'] == lr]
                lr_data = lr_data.sort_values('epochs')
                plt.plot(lr_data['epochs'], lr_data['test_precision'], marker='o', label=f"LR={lr}")

            plt.title(f"Hyperparameter Tuning: Precision@5 vs Epochs\n({experiment_name})")
            plt.xlabel("Epochs")
            plt.ylabel("Precision@5")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tuning_precision_{weight_key}_{feature_key}.png'))

            # Plot AUC
            plt.figure(figsize=(10, 6))
            for lr in sorted(results_df['learning_rate'].unique()):
                lr_data = results_df[results_df['learning_rate'] == lr]
                lr_data = lr_data.sort_values('epochs')
                plt.plot(lr_data['epochs'], lr_data['test_auc'], marker='o', label=f"LR={lr}")

            plt.title(f"Hyperparameter Tuning: AUC vs Epochs\n({experiment_name})")
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tuning_auc_{weight_key}_{feature_key}.png'))

            # Save results to CSV
            results_df.to_csv(os.path.join(output_dir, f'tuning_results_{weight_key}_{feature_key}.csv'), index=False)

            # Store results for this feature configuration
            all_results[feature_key] = {
                'best_params': {
                    'epochs': int(best_result['epochs']),
                    'learning_rate': float(best_result['learning_rate'])
                },
                'results_df': results_df,
                'feature_config': feature_config['name'],
                'metrics': {
                    'test_auc': float(best_result['test_auc']),
                    'test_precision': float(best_result['test_precision']),
                    'test_recall': float(best_result['test_recall'])
                }
            }

            # Store best parameters for this feature configuration
            best_params_overall[feature_key] = {
                'epochs': int(best_result['epochs']),
                'learning_rate': float(best_result['learning_rate']),
                'test_auc': float(best_result['test_auc'])
            }

        # Create a combined summary DataFrame of best configurations
        summary_data = []
        for feature_key, params in best_params_overall.items():
            summary_data.append({
                'Feature Config': feature_combinations[feature_key]['name'],
                'Feature Key': feature_key,
                'Best Epochs': params['epochs'],
                'Best Learning Rate': params['learning_rate'],
                'Test AUC': params['test_auc']
            })

        summary_df = pd.DataFrame(summary_data)

        return {
            'weight_type': weight_config['name'],
            'best_params_by_feature': best_params_overall,
            'all_results': all_results,
            'summary_df': summary_df
        }

    def saveBestPerformingModel(self, weight_key="3_naplan_weights", feature_keys=None):

        from lightfm import LightFM

        # Set default feature keys if not provided
        if feature_keys is None:
            feature_keys = ["5_no_user_feature_item_category"]
        elif isinstance(feature_keys, str):
            feature_keys = [feature_keys]  # Convert single string to list


        # Define feature combinations
        feature_combinations = {
            "1_no_features": {
                "user_features": None,
                "item_features": None,
                "name": "No Features"
            },
            "2_user_naplan": {
                "user_features": self.user_features,
                "item_features": None,
                "name": "NAPLAN User Features"
            },
            "3_user_naplan_item_category": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id"],
                "name": "NAPLAN + App Category"
            },
            "4_user_naplan_item_all": {
                "user_features": self.user_features,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "NAPLAN + App Category + Rating"
            },
            "5_no_user_feature_item_category": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id"],
                "name": "No User Feature + App Category"
            },
            "6_no_user_feature_item_all": {
                "user_features": None,
                "item_features": self.item_feature_configs["category_id_app_rating"],
                "name": "No User Feature + App Category + Rating"
            }
        }

        # Define weight types
        weight_types = {
            "1_no_weights": {
                "interactions": self.interaction_matrices["no_weights"],
                "weights": None,
                "name": "No Weights"
            },
            "2_duration_weights": {
                "interactions": self.interaction_matrices["duration_weights"],
                "weights": self.weights_matrices["duration_weights"], # Duration weights
                "name": "Duration Weights"
            },
            "3_naplan_weights": {
                "interactions": self.interaction_matrices["naplan_weights"],
                "weights": self.weights_matrices["naplan_weights"], # NAPLAN weights
                "name": "NAPLAN Weights"
            }
        }

        # Get the weight configuration
        weight_config = weight_types[weight_key]

        # Train model
        recommender_model = LightFM(
            loss="warp",
            learning_schedule="adadelta",
            learning_rate=0.001,
            random_state=42,
            user_alpha=0.01, item_alpha=0.01
        )

        feature_config = feature_combinations['5_no_user_feature_item_category']

        recommender_model.fit(
            weight_config['interactions'],
            sample_weight=weight_config['weights'],
            user_features=feature_config['user_features'],
            item_features=feature_config['item_features'],
            epochs=200,
            num_threads=4
        )

        # Save the model
        model_filename = 'lightfm_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(recommender_model, f)
        print(f"Model saved to {model_filename}")

        # Optional: Save with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f'lightfm_model_{timestamp}.pkl'
        with open(versioned_filename, 'wb') as f:
            pickle.dump(recommender_model, f)
        print(f"Versioned model saved to {versioned_filename}")

        # Save model configuration (without the sparse matrices)
        model_config = {
            'loss': 'warp',
            'learning_schedule': 'adadelta',
            'learning_rate': 0.001,
            'random_state': 42,
            'user_alpha': 0.01,
            'item_alpha': 0.01,
            'epochs': 200,
            'feature_config_name': '5_no_user_feature_item_category',  # Just save the name
            'user_features_shape': feature_config['user_features'].shape if feature_config['user_features'] is not None else None,
            'item_features_shape': feature_config['item_features'].shape if feature_config['item_features'] is not None else None,
            'training_date': timestamp
        }

        with open(f'model_config_{timestamp}.json', 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"Model configuration saved")

        # Save all training components together (including the actual features)
        training_package = {
            'model': recommender_model,
            'interactions': weight_config['interactions'],
            'weights': weight_config['weights'],
            'user_features': feature_config['user_features'],
            'item_features': feature_config['item_features'],
            'feature_config_name': '5_no_user_feature_item_category',
            'config': model_config
        }

        with open(f'complete_model_package_{timestamp}.pkl', 'wb') as f:
            pickle.dump(training_package, f)
        print(f"Complete training package saved")

        # Alternative: Save components separately for easier management
        # Save just the model
        with open(f'model_{timestamp}.pkl', 'wb') as f:
            pickle.dump(recommender_model, f)

        # Save features separately
        with open(f'features_{timestamp}.pkl', 'wb') as f:
            pickle.dump({
                'user_features': feature_config['user_features'],
                'item_features': feature_config['item_features'],
                'feature_config_name': '5_no_user_feature_item_category'
            }, f)

        # Save the JSON-compatible config
        with open(f'config_{timestamp}.json', 'w') as f:
            json.dump(model_config, f, indent=4)

        print(f"All components saved separately with timestamp: {timestamp}")

if __name__ == "__main__":
    # Initialize preprocessor
    olpc_model = OLPCLightFM()

    # Run full preprocessing pipeline
    data = olpc_model.preprocess_data()

    # Run hyperparameter tuning for a specific experiment configuration
    best_params_exp1 = olpc_model.tune_hyperparameters(
        weight_key="1_no_weights",
        feature_keys=["1_no_features", "2_user_naplan", "3_user_naplan_item_category", "4_user_naplan_item_all"],
        epochs_list=[10, 30, 50, 70, 100, 120, 150, 200, 250, 300],
        lr_list=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    )

    print(best_params_exp1)

    best_params_exp2 = olpc_model.tune_hyperparameters(
        weight_key="3_naplan_weights",
        feature_keys=["1_no_features", "5_no_user_feature_item_category", "6_no_user_feature_item_all"],
        epochs_list=[10, 30, 50, 70, 100, 120, 150, 200, 250, 300],
        lr_list=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    )

    print(best_params_exp2)

    best_params_exp3 = olpc_model.tune_hyperparameters(
        weight_key="2_duration_weights",
        feature_keys=["1_no_features", "2_user_naplan", "3_user_naplan_item_category", "4_user_naplan_item_all"],
        epochs_list=[10, 30, 50, 70, 100, 120, 150, 200, 250, 300],
        lr_list=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    )

    print(best_params_exp3)

    olpc_model.saveBestPerformingModel()