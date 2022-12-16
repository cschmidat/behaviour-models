from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=['vec'])
def hebb_update(w: jnp.ndarray, X: jnp.ndarray, y_hat: jnp.ndarray, eps: float, lam: float, vec = False) -> jnp.ndarray:
    """
    Performs one Hebbian update step.
    :param w: Weight
    :param X: Data matrix
    :param y_hat: Output
    :param eps: Learning rate
    :param lam: Weight decay constant
    :param vec: True if y_hat is a vector, False if it's a scalar
    :return: Calculated weight update Delta_W
    """
    if vec:
        deltaw = eps * (jnp.outer((y_hat), X) - lam * jnp.linalg.norm(y_hat)**2 * w) #Hebbian learning rule
        return deltaw
    else:
        deltaw = eps * ((X * (2*y_hat-1)) - lam * jnp.linalg.norm(2*y_hat-1)**2 * w).T #Hebbian learning rule
        return deltaw

def fsm_step(x: jnp.ndarray, minv: jnp.ndarray, w: jnp.ndarray, lam: float, eps: float) -> Tuple[jnp.ndarray]:
    """
    Performs one FSM step.
    :param x: Input
    :param minv: Lateral connections in hidden layer
    :param w: Connections to hidden layer
    :param lam: Weight decay constant
    :param eps: Learning rate
    :return: Calculated new weights minv, w
    """
    x = x.flatten()
    y = jnp.dot(minv, w.dot(x))
    delta_w = jnp.outer(2*eps * y, x)

    #Original weight decay formula:
    # w = (1 - 2 * self.pars.lr_fsm_w) * w + delta_w - w * self.pars.lam_fsm_w

    w = w + delta_w - lam*w * 2*eps

    # step = 2*eps
    # z = minv.dot(y)
    # c = step / (1 + step * jnp.dot(z, y))
    # minv = minv - jnp.outer(c * z, z.T) - minv * step
    m = jnp.linalg.inv(minv)
    minv = jnp.linalg.inv(m + eps*(jnp.outer(y, y)-lam*m))
    return minv, w
