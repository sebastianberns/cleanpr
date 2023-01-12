#!/usr/bin/env python

from collections import namedtuple
from pathlib import Path
from typing import Union, Optional
import warnings

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset

from cleanfeatures import CleanFeatures  # type: ignore[import]


Manifold = namedtuple('Manifold', ['features', 'radii'])


class PR:
    """
    Compute data features given an input source
        k (int): k-nearest neighbor parameter
        model_path (str, Path):  path to the save directory of embedding model checkpoints. Default: './models'
        model (str, optional): name of embedding model. Default: 'InceptionV3'
        cf (CleanFeatures, optional): instance of CleanFeatures
        device (str, device, optional):  device (e.g. 'cpu' or 'cuda:0')
        kwargs (dict): additional model-specific arguments passed on to CleanFeatures.
    """
    def __init__(self, k: int = 3,
                 model_path: Union[str, Path] = './models', model: str = 'InceptionV3', 
                 cf: Optional[CleanFeatures] = None,
                 device: Optional[Union[str, torch.device]] = None, **kwargs) -> None:
        self.k = k

        # Initialize clean features
        if cf is None:
            cf = CleanFeatures(model_path, model=model, device=device, log='warning', **kwargs)
        self.cf = cf

        # Initialize data manifold for later reference
        self.data_manifold: Optional[Manifold] = None


    """
    Compute precision and recall for given set of samples
        input (Tensor, Module, Dataset): set of samples to evaluate [N x D]
        kwargs (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method
    Return tuple of two floats: precision and recall
    """
    def precision_recall(self, input, **kwargs):
        assert self.data_manifold is not None, "Reference data manifold not available, first call 'set_data_manifold()'"

        input_manifold = self.get_manifold(input, **kwargs)

        precision = self.compute_metric(self.data_manifold, input_manifold)
        recall = self.compute_metric(input_manifold, self.data_manifold)
        return precision, recall

    __call__ = precision_recall


    """
    Build manifold from data source (tensor of samples, generator or dataset)
        input (Tensor, nn.Module, Dataset):  data source to process
        kwargs (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method
    Return manifold object with attributes 'features' (ndarray [N x D]) and 'radii' (ndarray [N])
    """
    def get_manifold(self, input: Union[Tensor, nn.Module, Dataset], **kwargs) -> Manifold:
        features = self.compute_features(input, **kwargs)
        distances = self.compute_pairwise_distances(features)
        radii = self.distances2radii(distances, k=self.k)
        return Manifold(features, radii)


    """
    Build and save data manifold for reference
        input (Tensor, nn.Module, Dataset):  data source to process
        kwargs (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method
    """
    def set_data_manifold(self, input: Union[Tensor, nn.Module, Dataset], **kwargs) -> None:
        self.data_manifold = self.get_manifold(input, **kwargs)


    """
    Compute features given a data source
        input (Tensor, nn.Module, Dataset):  data source to process
        kwargs (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method
    Return matrix of data features (ndarray) where rows are observations and columns are variables
    """
    def compute_features(self, input: Union[Tensor, nn.Module, Dataset], **kwargs) -> np.ndarray:
        if isinstance(input, Tensor):  # Tensor ready for processing
            features = self.cf.compute_features_from_samples(input, **kwargs)
        elif isinstance(input, nn.Module):  # Generator model
            features = self.cf.compute_features_from_generator(input, **kwargs)
        elif isinstance(input, Dataset):  # Data set
            features, targets = self.cf.compute_features_from_dataset(input, **kwargs)
        else:
            raise ValueError(f"Input type {type(input)} is not supported")
        
        return features.cpu().numpy()


    """
    Compute Precision and Recall metric
        manifold (Manifold): reference set of samples to test against
        subjects (Manifold): set of samples to evaluate
            For precision, 'manifold' is the data set and 'subjects' the generated samples
            For recall, 'manifold' is the generated samples and 'subjects' the data set
    Return ratio of subject samples that are covered by the manifold relative to the total number of samples
    """
    def compute_metric(self, manifold: Manifold, subjects: Manifold) -> float:
        N = subjects.features.shape[0]  # Number of items
        count = 0  # Counter
        dist = self.compute_pairwise_distances(manifold.features, subjects.features)  # Pair-wise distances
        for i in range(N):  # For all items
            count += (dist[:, i] < manifold.radii).any()  # Is the item within the radius of any other item?
        coverage = count / N  # Coverage ratio: amount of items that fall into the manifold
        return coverage


    """
    Compute pairwise distances between two collections of items or between the items of one collection
        X (ndarray): feature matrix [N x D]
        Y (ndarray): optional second feature matrix [N x D]
    Return distance matrix (symmetric ndarray [N x N])
    """
    def compute_pairwise_distances(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = X

        # Check basic assumptions
        assert len(X.shape) == 2  # Is matrix
        assert len(Y.shape) == 2
        assert X.shape[0] == Y.shape[0]  # Same size
        assert X.shape[1] == Y.shape[1]

        # Helper function
        def squared_norm(A):
            num = A.shape[0]
            A = A.astype(np.float64)  # to prevent underflow
            A = np.sum(A**2, axis=1, keepdims=True)
            A = np.repeat(A, num, axis=1)
            return A

        # Squared norms
        X2 = squared_norm(X)
        Y2 = squared_norm(Y).T

        XY = np.dot(X, Y.T)  # Gram matrix
        D2 = X2 - 2*XY + Y2  # Euclidean distance matrix

        # check negative distance
        negative = D2 < 0  # Indices of entries below zero
        if negative.any():  # Are there any negative squared distances?
            D2[negative] = 0.  # Set to zero
            warnings.warn(f"{negative.sum()} negative squared distances found and set to zero")

        distances = np.sqrt(D2)  # Actual distances
        return distances


    """
    Convert pairwise distances to radii
        distances (ndarray [N x N]): pairwise distance matrix
        k (int): k-nearest neighbor parameter
    Return vector of radii (ndarray [N])
    """
    def distances2radii(self, distances: np.ndarray, k: int) -> np.ndarray:
        num_features = distances.shape[0]
        radii = np.zeros(num_features)
        for i in range(num_features):
            radii[i] = self._get_kth_value(distances[i], k=k)
        return radii


    """ 
    Calculate radius around individual item based on k nearest neighbors
    Helper function to distances2radii
        distances (ndarray [N]): distances of one item to all other items
        k (int): k-nearest neighbor parameter
    Return radius around item based on k
    """
    def _get_kth_value(self, distances: np.ndarray, k: int) -> float:
        k_ = k + 1  # kth NN should be (k+1)th because closest one is itself
        idx = np.argpartition(distances, k_)  # Partial sort, get indices
        k_closest = distances[idx[:k_]]  # Get k smallest distances to other items
        kth_value = k_closest.max()  # Get biggest distance to k closest items
        return kth_value
