import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(69)

def plot_w_genre(genres, movie_df, V, dim_3=False, 
                 n_mov=30, colors=None, ax=None,
                 annotate=True):
    '''
    Plot V's features, in 2d or 3d (set dim_3 to True for 3d),
    separating colors based on genre
    
    Args:
        genres: list of genres to consider
        movie_df: a movie dataframe with info on
                what genres each movie is
        V: a matrix of (nMovies, k) dimensions, (result of
            using PCA / SVD on V from collaborative filtering),
            principal components to plot
        n_mov: number of movies of each genre to plot,
        colors: List of colors of same length as genres list
                (or leave as None)
        ax: An axis to plot on, has to be a 3d axis if dim_3 is True
        annotate: Whether to label points with movie titles or not
        dim_3: Plot in 3d (default is 2d). 
               **N.B. from mpl_toolkits.mplot3d import Axes3D must be added**
               
    Returns:
        axis if ax was None
    '''
    assert(V.shape[1] >= (3 if dim_3 else 2))
    ids = [movie_df.loc[movie_df[g] == 1, "Movie Id"][:n_mov].values for g in genres]
    return plot_w_ids(ids, movie_df, V, dim_3=dim_3, 
                      n_mov=n_mov, colors=colors, ax=ax,
                      annotate=annotate, labels=genres)
    
    
def plot_w_ids(ids, movie_df, V, dim_3=False, 
                 n_mov=30, colors=None, ax=None,
                 annotate=True, labels=None):
    '''
    Plot V's features, in 2d or 3d (set dim_3 to True for 3d),
    separating colors based on groups of ids
    
    Args:
        ids: list of np arrays of ids, (list of groups), each group is a color
        movie_df: a movie dataframe with info on
                what each movie is
        V: a matrix of (nMovies, k) dimensions, (result of
            using PCA / SVD on V from collaborative filtering),
            principal components to plot
        n_mov: number of movies of each genre to plot,
        colors: List of colors of same length as ids list
                (or leave as None)
        labels: List of names of same length as ids is
        ax: An axis to plot on, has to be a 3d axis if dim_3 is True
        annotate: Whether to label points with movie titles or not
        dim_3: Plot in 3d (default is 2d). 
               **N.B. from mpl_toolkits.mplot3d import Axes3D must be added**
               
    Returns:
        axis if ax was None
    '''
    assert(V.shape[1] >= (3 if dim_3 else 2))
    sub_Vs = [V[sub_ids - 1] for sub_ids in ids]

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(ids)))
    if ax is None:
        axis = plt.figure().add_subplot(111, projection='3d') if dim_3 else plt.subplot(111)
    else:
        axis = ax
    if labels is None:
        labels = [f"Group {i}" for i in range(len(ids))]

    for c, sub_V, sub_ids in zip(colors, sub_Vs, ids):
        # Handle 3d case
        if dim_3:
            axis.scatter(sub_V[:, 0], sub_V[:, 1], sub_V[:, 2], color=c)
            if annotate:
                [axis.text(*x[:3], movie_df["Movie Title"].values[Id]) for (Id, x) in zip(sub_ids, sub_V)]
            continue
                
        axis.scatter(sub_V[:, 0], sub_V[:, 1], color = c)
        if annotate:
            [axis.annotate(movie_df["Movie Title"].values[Id], x[:2]) for (Id, x) in zip(sub_ids, sub_V)]
            
    axis.legend(labels)
    if ax is None:
        return axis

def best_movs_ids(data_df, nMovies=10, get_worst=False):
    '''
    Get the ids of the best movies (highest av rating) 
    from the data df
    '''
    av_ratings = data_df.groupby("Movie Id").aggregate('mean')
    sort_ratings = av_ratings.sort_values(by='Rating', axis='index', ascending=get_worst)
    return sort_ratings[:nMovies].index

def pop_movs_ids(data_df, nMovies=10, get_least_pop=False):
    '''
    Get the ids of the most popular movies (most ratings) 
    from the data df
    '''
    direc = -1 if get_least_pop else 1
    val_counts = data_df["Movie Id"].value_counts(sort=True)[::direc]
    return val_counts[:nMovies].index

def subselect_from_ids(data_df, ids, V = None):
    '''
    Subselect rows from either the data df (if V is None)
    or V matrix, given a list of ids to use
    '''
    inds = data_df["Movie Id"].isin(ids)
    if V is None:
        return data_df.loc[inds - 1, :]
    return V[inds]

def get_genre_reps(V, movie_df, normalize=False):
    inds = movie_df[genre] >= 0
    V[inds, :]
    
    
    
    