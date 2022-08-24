from typing import List, Union
import jax
import jax.numpy as jnp


def gen_data_x(key: jnp.ndarray, n_points: int, n_dim: int, n_classes: int, sigmas: Union[List[float],jnp.ndarray], means: jnp.ndarray = None, vec=False):
    """
    Generate random sample data from multivariate normal distributions.
    :param key: PRNG key
    :param n_points: number of sample points
    :param n_dim: number of dimensions
    :param n_classes: number of classes
    :param sigmas: list of standard deviations or array of covariances
    :param means: Array of means
    :param vec: True if sigmas are scalar s.d., False if they are covariances
    :return:
        X: data matrix
        y: label vector
        means: vector of means"""
    
    
    if vec:
        y = jax.random.randint(key, (n_points,), 0, n_classes)
        X_centered = jnp.einsum('ijk,ik->ij',jnp.linalg.cholesky(sigmas)[y], jax.random.normal(key, shape=(n_points, n_dim)), optimize='optimal')
        X = X_centered + means[y]
    else:
        assert n_classes == len(sigmas)
        y = jax.random.randint(key, (n_points,), 0, n_classes)
        X = sigmas[y][:, None] * jax.random.normal(key, shape=(n_points, n_dim)) + means[y]
    return X, y, means
    