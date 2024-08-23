import torch
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_movielens_dataframes():
    """
    Load the movielens/latest-small-ratings and movielens/latest-small-movies datasets from TFDS.

    Returns:
        ratings_df (pd.DataFrame): DataFrame containing the user ratings data.
        movies_df (pd.DataFrame): DataFrame containing the movie details.

    Example:
        ratings_df, movies_df = load_movielens_dataframes()
    """

    # Load the data
    ratings = tfds.load('movielens/latest-small-ratings', split="train", download=True)
    movies = tfds.load('movielens/latest-small-movies', split="train", download=True)

    # Convert ratings data to DataFrame
    ratings_records = [{
        'user_id': example['user_id'].numpy().decode('utf-8'),
        'movie_id': example['movie_id'].numpy().decode('utf-8'),
        'user_rating': example['user_rating'].numpy()
    } for example in ratings]
    ratings_df = pd.DataFrame(ratings_records)

    # Convert movies data to DataFrame
    movies_records = [{
        'movie_title': example['movie_title'].numpy().decode('utf-8'),
        'movie_id': example['movie_id'].numpy().decode('utf-8'),
        'movie_genres': list(example['movie_genres'].numpy()) # already encoded (e.g., 1 might correspond to "Action", 2 to "Comedy", and so forth)
    } for example in movies]
    movies_df = pd.DataFrame(movies_records)

    return ratings_df, movies_df


class GraphDataPreparation:
    def __init__(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, split = 0.2
    ):
        """
        Initialize the class with ratings and movies dataframes.

        Args:
            ratings_df: DataFrame with ratings data.
            movies_df: DataFrame with movies data.
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        # Call the function to remove movies without ratings
        self.filter_movies_with_ratings()
        # Create mappings for userId and movieId to new continuous indexes
        self.user_mapping = {
            user_id: i for i, user_id in enumerate(self.ratings_df.user_id.unique())
        }
        self.movie_mapping = {
            movie_id: i for i, movie_id in enumerate(self.ratings_df.movie_id.unique())
        }
        # Apply the mappings to set new continuous userIds and movieIds
        self.ratings_df["user_id"] = self.ratings_df["user_id"].map(self.user_mapping)
        self.ratings_df["movie_id"] = self.ratings_df["movie_id"].map(
            self.movie_mapping
        )
        self.movies_df["movie_id"] = self.movies_df["movie_id"].map(self.movie_mapping)
        # Store movie ID to title mapping
        self.movie_id_to_title = dict(zip(self.movies_df['movie_id'], self.movies_df['movie_title']))
        # Initialize edge splitting transformation
        self.transform = RandomLinkSplit(is_undirected=False, num_val=split)

    def filter_movies_with_ratings(self):
        # Extract unique movie IDs from ratings_df
        unique_movie_ids_in_ratings = set(self.ratings_df['movie_id'].unique())

        # Filter out movies in movies_df that don't have a corresponding rating in ratings_df
        initial_count = len(self.movies_df)
        self.movies_df = self.movies_df[self.movies_df['movie_id'].isin(unique_movie_ids_in_ratings)].copy()
        removed_count = initial_count - len(self.movies_df)

        if removed_count > 0:
            print(f"Removed {removed_count} movies from movies_df as they lack corresponding ratings.")
        else:
            print("No movies were removed. All movie IDs in movies_df have ratings in ratings_df.")

    def create_edge_index(self):
        """
        Create an edge index for the graph. Edge direction is from user to movie.

        Returns:
            A tensor representing the edge index.
        """
        user_nodes = self.ratings_df["user_id"].to_numpy()
        movie_nodes = (
            self.ratings_df["movie_id"].to_numpy()
            + self.ratings_df["user_id"].nunique()
        )

        edge_index = torch.tensor(np.array([user_nodes, movie_nodes]), dtype=torch.long)

        return edge_index

    def create_edge_features(self):
        """
        Create edge features for the graph.

        Returns:
            A tensor representing the edge features.
        """
        ratings = self.ratings_df["user_rating"].to_numpy()
        edge_attr = torch.tensor(ratings, dtype=torch.float).view(-1, 1)

        return edge_attr

    def create_node_features(self):
        """
        Create node features for the graph.

        Returns:
            A tensor representing the node features.
        """

        # Prepare movie features
        movie_genres = self.movies_df["movie_genres"].apply(lambda x: pd.Series(x))
        movie_features = (
            pd.get_dummies(movie_genres.stack()).groupby(level=0).sum().values
        )

        # Prepare user features
        self.num_users = self.ratings_df["user_id"].nunique()
        user_features = np.zeros((self.num_users, movie_features.shape[1])) # TODO: this is initialized to zeros, but updated within the model

        # Combine user and movie features
        node_features = np.vstack([user_features, movie_features])
        node_features = torch.tensor(node_features, dtype=torch.float)

        return node_features

    def create_edge_mask(self, original_edge_index, edge_label_index):
        # Convert the edges to a set of tuples for easy lookup
        label_edges_set = set([tuple(x) for x in edge_label_index.t().numpy()])

        # Generate the mask by checking each edge in original_edge_index
        mask = [tuple(x) for x in original_edge_index.t().numpy()]
        mask = [edge in label_edges_set for edge in mask]

        return np.array(mask)[:, None]

    def prepare_data(self):
        """
        Prepare the graph data.

        Returns:
            A train and test PyG Data object with the prepared graph data.
        """
        edge_index = self.create_edge_index()
        edge_attr = self.create_edge_features()
        node_features = self.create_node_features()

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        train_graph, val_graph, test_graph = self.transform(graph)

        train_mask = self.create_edge_mask(graph.edge_index, train_graph.edge_label_index)
        val_mask = self.create_edge_mask(graph.edge_index, val_graph.edge_label_index)
        test_mask = self.create_edge_mask(graph.edge_index, test_graph.edge_label_index)

        graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask
        graph.num_users = self.num_users
        graph.movie_id_to_title = self.movie_id_to_title

        return graph


if __name__ == "__main__":

    ratings_df, movies_df = load_movielens_dataframes()
    graph_prep = GraphDataPreparation(ratings_df=ratings_df.copy(), movies_df=movies_df.copy(), split=0.4)
    graph = graph_prep.prepare_data()
    torch.save(graph, "practicals/data.pt")
    print("+++++++ DONE ++++++++++++")