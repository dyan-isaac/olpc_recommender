import numpy as np
from lightfm import LightFM
import pickle


class AppRecommender:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.user_features = None
        self.item_features = None
        self.sample_weight = None

    def train(self, interactions, dataset, user_features=None, item_features=None, epochs=50, learning_rate=0.02,
              sample_weight=None):
        """
        Train the recommendation model with the best parameters.

        Parameters:
        -----------
        interactions : scipy.sparse matrix
            The interaction matrix
        dataset : lightfm.data.Dataset
            The LightFM dataset object containing mappings
        user_features : scipy.sparse matrix, optional
            User features matrix
        item_features : scipy.sparse matrix, optional
            Item features matrix
        epochs : int, default=50
            Number of epochs to train
        learning_rate : float, default=0.02
            Learning rate for training
        components : int, default=64
            Number of latent components
            :param sample_weight:
        """
        # Store dataset for later use in recommendations
        self.dataset = dataset
        self.user_features = user_features
        self.item_features = item_features
        self.sample_weight = sample_weight

        # Create model with best parameters
        self.model = LightFM(
            learning_schedule='adagrad',
            loss='warp',
            user_alpha=0.01,
            item_alpha=0.01,
            learning_rate=learning_rate,
            random_state=42
        )

        # Train the model
        self.model.fit(
            interactions=interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=epochs,
            verbose=False,
            sample_weight=sample_weight,
            num_threads=4
        )

        print(f"Model trained - {epochs} epochs!")

    def recommend(self, school_id, interaction_df, n=10, exclude_seen=True, include_app_details=True):
        """
            Recommend apps for a specific school.
            """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert school_id to int if it's not already
        school_id = int(school_id)

        # Get user and item mappings
        mappings = self.dataset.mapping()

        # Handle different possible mapping structures
        if isinstance(mappings, tuple) and len(mappings) >= 2:
            user_mapping = mappings[0]
            item_mapping = mappings[1]
        else:
            user_mapping = mappings.get('user_id_map', {})
            item_mapping = mappings.get('item_id_map', {})

        # Debug information
        print(f"Looking for school ID {school_id}")
        print(f"User mapping type: {type(user_mapping)}")
        print(f"First few keys in user mapping: {list(user_mapping.keys())[:5]}")

        # Check if school exists in the dataset
        if school_id not in user_mapping:
            # Try string conversion as a fallback
            if str(school_id) in user_mapping:
                school_id = str(school_id)
            else:
                raise ValueError(f"School ID {school_id} not found in the dataset")

        # Get internal ID for the school
        school_idx = user_mapping[school_id]

        # Get all app IDs and their indices
        app_ids = list(item_mapping.keys())
        app_indices = [item_mapping[app_id] for app_id in app_ids]

        # Get apps already used by the school if excluding seen apps
        if exclude_seen:
            seen_apps = set(interaction_df[interaction_df['school_id'] == school_id]['app_id'])
            unseen_apps = [app_id for app_id in app_ids if app_id not in seen_apps]

            if not unseen_apps:
                print(
                    f"Warning: School {school_id} has interacted with all available apps. Returning recommendations from all apps.")
                unseen_apps = app_ids

            unseen_indices = [item_mapping[app_id] for app_id in unseen_apps]
        else:
            unseen_apps = app_ids
            unseen_indices = app_indices

        # Predict scores for all unseen apps
        scores = self.model.predict(
            user_ids=[school_idx] * len(unseen_indices),
            item_ids=unseen_indices,
            user_features=self.user_features,
            item_features=self.item_features
        )

        # Create a list of (app_id, score) tuples
        app_scores = list(zip(unseen_apps, scores))

        # Sort by score (descending) and get top n
        top_recommendations = sorted(app_scores, key=lambda x: x[1], reverse=True)[:n]

        # If details are requested, add category and rating information
        if include_app_details:
            detailed_recommendations = []
            for app_id, score in top_recommendations:
                # Find this app in the dataframe
                app_info = interaction_df[interaction_df['app_id'] == app_id].iloc[0]

                detailed_recommendations.append({
                    'app_id': app_id,
                    'score': float(score),  # Convert numpy float to Python float
                    'category_id': app_info['category_id'],
                    'app_rating': float(app_info['app_rating']) if 'app_rating' in app_info else None
                })
            return detailed_recommendations
        else:
            return top_recommendations

    def recommend_for_all_schools(self, interaction_df, n=5, exclude_seen=True):
        """
        Generate recommendations for all schools in the interaction dataframe.

        Parameters:
        -----------
        interaction_df : pandas DataFrame
            DataFrame containing the interactions data
        n : int, default=5
            Number of recommendations per school
        exclude_seen : bool, default=True
            Whether to exclude apps each school has already interacted with

        Returns:
        --------
        dict
            A dictionary mapping school IDs to their recommendations
        """
        all_recommendations = {}
        for school_id in interaction_df['school_id'].unique():
            try:
                recommendations = self.recommend(
                    school_id=school_id,
                    interaction_df=interaction_df,
                    n=n,
                    exclude_seen=exclude_seen
                )
                all_recommendations[school_id] = recommendations
            except Exception as e:
                print(f"Error recommending for school {school_id}: {str(e)}")

        return all_recommendations

    def get_similar_apps(self, app_id, n=10):
        """
        Find apps similar to a given app based on the learned item embeddings.

        Parameters:
        -----------
        app_id : str or int
            ID of the app to find similar items for
        n : int, default=10
            Number of similar apps to return

        Returns:
        --------
        list of tuples
            (app_id, similarity_score) tuples for the most similar apps
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get internal mappings
        _, item_mapping = self.dataset.mapping()

        # Check if app exists
        if app_id not in item_mapping:
            raise ValueError(f"App ID {app_id} not found in the dataset")

        # Get the embedding for this app
        app_idx = item_mapping[app_id]
        app_embedding = self.model.item_embeddings[app_idx]

        # Calculate similarity with all other apps
        similarities = []
        for other_app_id, other_idx in item_mapping.items():
            if other_app_id == app_id:
                continue

            other_embedding = self.model.item_embeddings[other_idx]

            # Cosine similarity
            similarity = np.dot(app_embedding, other_embedding) / (
                    np.linalg.norm(app_embedding) * np.linalg.norm(other_embedding)
            )

            similarities.append((other_app_id, float(similarity)))

        # Sort by similarity (descending) and return top n
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    def save(self, filename):
        """Save the trained model and related data to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load a trained model from a file."""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

""""
# Later, in another script
loaded_recommender = AppRecommender.load("app_recommender_model.pkl")
"""