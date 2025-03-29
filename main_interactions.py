import os
import pandas as pd
import numpy as np
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from experiments.experiment1 import run_experiment1, run_experiment1_interaction
from experiments.experiment2 import run_experiment2, run_experiment2_interaction
from experiments.experiment3 import run_experiment3, run_experiment3_interaction

KAGGLE_DIR = "/kaggle/input/olpc-log/"
LOCAL_DIR = "data"

def _get_data_path():
    """
    Get the path to your OLPC dataset files.
    """
    return os.path.join(os.getcwd(), LOCAL_DIR)

def load_naplan_scores():
    """
    Load naplan_scores.csv and compute average or bin them, etc.
    """
    print("1. Loading naplan_scores.csv...")
    performance_path = os.path.join(_get_data_path(), "naplan.csv")
    naplan_df = pd.read_csv(performance_path)

    # Calculate average score per school per year_tested
    yearly_avg = naplan_df.groupby(['school_id', 'year_tested'])['naplan_score'].mean().reset_index()

    # Then, calculate overall average per school
    school_avg_scores = yearly_avg.groupby('school_id')['naplan_score'].mean().reset_index()
    school_avg_scores.rename(columns={'naplan_score': 'avg_score'}, inplace=True)

    # Binning avg_score per school
    bins = [0, 400, 420, 440, 460, float('inf')]
    labels = ['bin1','bin2','bin3','bin4','bin5']

    print("1.1 Binning average scores to bins 1-5");
    school_avg_scores['score_bin'] = pd.cut(school_avg_scores['avg_score'], bins=bins, labels=labels, right=False)

    scaler = StandardScaler()
    school_avg_scores['norm_avg_score'] = scaler.fit_transform(school_avg_scores[['avg_score']])

    return school_avg_scores

def load_app_usage():
    """
    Load aggregated-school-device-app-duration.csv and do device count normalization, etc.
    """
    print("\n2. Loading aggregated app usage for school-device-app...")
    usage_path = os.path.join(_get_data_path(), "aggregated-school-device-app-category-rating-duration.csv")
    usage_df = pd.read_csv(usage_path)
    usage_df.rename(columns={'school id': 'school_id', 'app id': 'app_id', 'total duration': 'duration', 'device id': 'device_id', 'category id': 'category_id', 'app rating':'app_rating'}, inplace=True)

    return usage_df[['school_id', 'device_id', 'app_id', 'category_id', 'app_rating', 'duration']]

def normalize_app_usage(usage_df):
    """
    Normalize aggregated app usage duration per school-app pair by the number of devices per school.
    """
    print("\n3. Normalizing aggregated app usage duration per school-app pair...")
    devices_per_school = usage_df.groupby('school_id')['device_id'].nunique().reset_index().rename(columns={'device_id': 'device_count'})
    usage_df = usage_df.merge(devices_per_school, on='school_id', how='left')

    usage_agg = usage_df.groupby(['school_id', 'app_id', 'category_id', 'app_rating', 'device_count']).agg({'duration': 'sum'}).reset_index()

    print("3.1 Removing app duration that yield to 0 app usage duration.")
    # Only keep app usage when duration is not zero because this indicates no actual interaction.
    # Keeping them clutters the dataset with explicit zeros that LightFM already assumes implicitly for all
    # non-observed user–item pairs.
    count_zero_durations = (usage_agg['duration'] == 0).sum()
    print(f"    Number of rows with zero duration to be removed: {count_zero_durations}")
    usage_agg = usage_agg[usage_agg['duration'] > 0]

    print("3.2 Penalising duration with number of devices per school app usage.")
    usage_agg['duration_device_ratio'] = usage_agg['duration'] / usage_agg['device_count']

    print("3.3 Performing log transformation for duration_device_ratio")
    usage_agg['norm_duration_device_ratio'] = np.log1p(usage_agg['duration_device_ratio'])

    print("3.4 After log transform, data is still positively skewed, hence we use quantile transformation to have a normal distribution.")
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    usage_agg['scaled_duration'] = transformer.fit_transform(usage_agg[['norm_duration_device_ratio']])

    return usage_agg[['school_id', 'app_id', 'category_id','app_rating', 'scaled_duration']]

def merge_datasets(performance_df, usage_df):
    """
    Merge app usage data with school performance data, including avg_score.
    """
    merged_df = usage_df.merge(performance_df[['school_id', 'score_bin']], on='school_id', how='left')

    return merged_df

def build_dataset():
    """
    High-level function to return:
      - dataset (LightFM's Dataset object)
      - interactions, weights
      - user_features (if any)
      - train/test sets if you want to do the split here
    """
    # 1) Load data
    perf_df = load_naplan_scores()
    usage_df = load_app_usage()
    normalized_app_usage_df = normalize_app_usage(usage_df)

    print(f"\n4. Merge school performance data and app usage data.")
    # 2) Merge & preprocess
    merged_df = merge_datasets(perf_df, normalized_app_usage_df)

    count_null = merged_df['score_bin'].isna().sum()
    print(f"\n5. Delete number of NaN score_bin rows: {count_null}")

    interaction_df = merged_df[['school_id', 'app_id', 'category_id','app_rating', 'score_bin', 'scaled_duration']].copy()

    school_ids = interaction_df['school_id'].astype('category')
    app_ids = interaction_df['app_id'].astype('category')
    interaction_df['category_id'] = interaction_df['category_id'].astype(int)

    all_bins = ['bin1','bin2','bin3','bin4','bin5']
    user_feature_labels = [f"score_bin:{b}" for b in all_bins]

    unique_cats = interaction_df['category_id'].unique().tolist()  # now ints
    unique_ratings = interaction_df['app_rating'].unique().tolist()

    # Prepare labels for LightFM's dataset
    cat_feature_labels = [f"category_id:{cat}" for cat in unique_cats]
    rating_feature_labels = [f"app_rating:{rating}" for rating in unique_ratings]

    # Combine them so the dataset knows all possible item features
    item_feature_labels = cat_feature_labels + rating_feature_labels

    # Initialize and fit the Dataset with known user/item IDs and features
    dataset = Dataset()
    dataset.fit(
        users=school_ids,
        items=app_ids,
        user_features=user_feature_labels,
        item_features=item_feature_labels
    )

    # Build interaction matrices
    # LightFM returns: interactions (binary)
    (interactions, weights) = dataset.build_interactions(
        (
            (row.school_id, row.app_id)
            for _, row in interaction_df.iterrows()
        )
    )

    # Build user features matrix
    # - Each school has exactly one score_bin
    user_features_df = (
        interaction_df[['school_id', 'score_bin']]
        .drop_duplicates(subset='school_id')
        .dropna(subset=['score_bin'])  # ensure no missing bins
    )
    user_features = dataset.build_user_features(
        (
            (row.school_id, [f"score_bin:{row.score_bin}"])
            for _, row in user_features_df.iterrows()
        )
    )

    # Build item features matrix
    # - Each app has exactly one category_id and app_rating (assuming 1:1 relationship).
    item_features_df = (
        interaction_df[['app_id', 'category_id', 'app_rating']]
        .drop_duplicates(subset='app_id')
        .dropna(subset=['category_id'])
        .dropna(subset=['app_rating'])
    )

    item_feature_configs = {}
    for config in [['category_id'], ['category_id', 'app_rating']]:
        feature_cols = ['app_id'] + config
        temp_df = item_features_df[feature_cols].copy()
        if 'category_id' in config:
            temp_df['category_id'] = temp_df['category_id'].astype(int)
        if 'app_rating' in config:
            temp_df['app_rating'] = temp_df['app_rating'].round(1)
        feature_matrix = dataset.build_item_features(
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
        item_feature_configs[key] = feature_matrix

    return dataset, interactions, weights, user_features, item_feature_configs

def run_all_experiments():
    # build_dataset, do random_train_test_split, etc.
    dataset, interactions, weights, user_features, item_feature_configs = build_dataset()
    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(3))

    print("\n=== Running Experiment 1 ===")
    exp1_results = run_experiment1_interaction(train, test)
    print("-------------------------------------------------------------")
    print(f"{'Epochs':>6} | {'LR':>5} | {'Precision@5':>12} | {'AUC':>6}")
    print("-------------------------------------------------------------")

    for res in exp1_results:
        print(f"{res['epochs']:>6} | "
              f"{res['learning_rate']:>5} | "
              f"{res['precision']:.4f}{'':>6} | "
              f"{res['auc']:.4f}")


    print("\n=== Running Experiment 2 ===")
    exp1_results = run_experiment2_interaction(train, test, user_features)
    print("-------------------------------------------------------------")
    print(f"{'Epochs':>6} | {'LR':>5} | {'Precision@5':>12} | {'AUC':>6}")
    print("-------------------------------------------------------------")

    for res in exp1_results:
        print(f"{res['epochs']:>6} | "
              f"{res['learning_rate']:>5} | "
              f"{res['precision']:.4f}{'':>6} | "
              f"{res['auc']:.4f}")

    for config_key, item_features in item_feature_configs.items():
        print(f"\n=== Running Experiment 3 ({config_key}) ===")
        results = run_experiment3_interaction(train, test, user_features, item_features)
        print("-------------------------------------------------------------")
        print(f"{'Epochs':>6} | {'LR':>5} | {'Precision@5':>12} | {'AUC':>6}")
        print("-------------------------------------------------------------")
        for res in results:
            print(f"{res['epochs']:>6} | {res['learning_rate']:>5} | {res['precision']:.4f}{'':>6} | {res['auc']:.4f}")

if __name__ == "__main__":
    run_all_experiments()
