from typing import Callable
from hypo_x import hebb_update, fsm_step
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from data_gen_x import gen_data_x

"""
Condition Functions
"""

def an_cond(i, _):
    """Active nothing condition
    :param i: Array of indices.
    :return: Array of 'True' values"""
    return jnp.full_like(i, True, dtype=bool)


def ap_cond(i, _):
    """
    Active passive condition.
    :param i: Array of indices.
    :return: Array where every 10-th element is True,
        others are False.
    """
    return i % 10 == 0


def pta_cond(i, num_it):
    """
    Passive then active condition.
    :param i: Array of indices.
    :param num_it: Total number of iterations.
    :return: Array where indices corresponding to last 10%
        of trials are True.
    """
    return i >= num_it * 9 / 10



"""
Neural Network helper functions
"""


def ret_fun(x, y, weights):
    """Trivial helper function."""
    return weights



def loss(y_true: jnp.ndarray, y_pred:jnp.ndarray, weight: jnp.ndarray, lam: float=0):
    """
    Binary cross entropy loss plus l^2 regularization.
    :param y_true: True labels (0 or 1)
    :param y_pred: Predicted class probabilities (in [0,1])
    :param weight: Weight used for regularization
    :param lam: l^2 regularization parameter
    :return: (average) loss value
    """
    x_ent = - (y_true*jnp.log(y_pred) + (1-y_true)*jnp.log(1-y_pred))
    return lam * jnp.linalg.norm(weight)**2 + x_ent.mean()

def single_out(x: jnp.ndarray, v: jnp.ndarray):
    """
    Output for single layer network.
    :param x: Network input.
    :param v: Network weight."""
    return jax.nn.sigmoid(x @ v)


def double_out(x, v, minv, w):
    """
    Logit output function for two-layer network.
    :param x: Input
    :param v: Output weight
    :param minv: Lateral connections in hidden layer
        (only nontrivial for Similarity Matching)
    :param w: Connections to hidden layer
    """
    hid =  x @ w.T @ minv.T
    return jax.scipy.special.expit(hid @ v)

def double_out_logit(x, v, minv, w):
    """
    Logit output function for two-layer network.
    :param x: Input
    :param v: Output weight
    :param minv: Lateral connections in hidden layer
        (only nontrivial for Similarity Matching)
    :param w: Connections to hidden layer
    """
    hid = x @ w.T @ minv.T
    return hid @ v

def sigmoid_cross_entropy_with_logits(x, z, weight: jnp.ndarray, lam: float=0):
    """
    Parameters:
    x: logits
    z: labels
    """
    return lam * jnp.linalg.norm(weight)**2 + jnp.mean(jnp.maximum(x, 0) - x * z + jnp.log1p(jnp.exp(-jnp.abs(x))))

"""
Parameter classes
"""
def pancake_sigs(sigs, dim, scale):
    """
    Prepares covariance matrix for pancake-like distribution.
    :param sigs: List of stdard deviations
    :param dim: Dimension
    :param scale: Scale determining by how much distribution is widened
    :return: Diagonal covariance matrices with second component rescaled by 'scale'
    """
    sigs = [sig**2 for sig in sigs]
    sigmas = jnp.array([jnp.diag(jnp.array([sigs[0]] + [scale*sigs[0]] + [sigs[0]]*(dim-2))),
                        jnp.diag(jnp.array([sigs[1]] + [scale*sigs[1]] + [sigs[1]]*(dim-2)))])
    return sigmas

@dataclass
class parclass_base():
    """
    Base parameter class.
    """
    dim: int = 20
    pancake_scale: float = 2
    #Means should be override-able
    means: jnp.ndarray = field(default=jnp.array([[-0.1]+[0]*(dim-1), [0.1] + [0]*(dim-1)]), init=False)
    sigs: jnp.ndarray = pancake_sigs([0.07, 0.07], dim, scale=pancake_scale)
    num_psy_points: int = 6
    num_val_points: int = 1000
    debug: bool = False
    
@dataclass
class parclass_v(parclass_base):
    """
    Parameter class for one-layer network.
    """
    lr_sgd_v: float = 1e-2
    lr_hebb_v: float = 1e-2
    lam_sgd_v: float = 1e-1
    lam_hebb_v: float = 1e-1
    sup_v: bool = True #Supervised learning for v

@dataclass
class parclass_wv(parclass_base):
    """
    Parameter class for two-layer network.
    """
    dim_hid = 2
    lr_sgd_v: float = 1e-2
    lr_sgd_v_decay: float = 0
    lr_hebb_v: float = 1e-2
    lam_sgd_v: float = 1e-1
    lam_hebb_v: float = 1e-1
    lr_sgd_w: float = 1e-2
    lr_fsm_w: float = 1e-3
    lam_sgd_w: float = 1e-2
    lam_fsm_w: float = 1e-4
    sup_v: bool = True  #Supervised learning for v
    sup_w: bool = True  #Supervised learning for w
    unsup_v: bool = True #Unsupervised learning for v
    unsup_w: bool = True #Unsupervised learning for w
    hebb_w: bool = False




class BaseModel():
    """"
    Base model class for the networks we consider.
    This implements the training routines in JAX."""
    def __init__(self):
        raise NotImplementedError
     

    def update(self, weights, data):
        """
        Run one update step, corresponding to the specified training scheme.
        :param weights: (Tuple of) all weights for the model
        :param data: Array of packaged training data, with active and passive samples,
            and a Boolean specifying if the sample is trained with active learning or not.
        """
        x, x_pas, y, active_bool, i = data[:self.dim], data[self.dim:2*self.dim], data[-3], data[-2], data[-1]
        X_val, y_val = self.X_val, self.y_val
        #Perform active (and passive) fit if the trial is active, and a passive only fit otherwise.
        weights = jax.lax.cond(
            active_bool, self.active_passive_fit_one, self.passive_fit_one_psy, x, x_pas, y, weights, i)
        metric = self.keep_log(X_val, y_val, weights)
        return weights, metric
    def active_passive_fit_one(self, x, x_pas, y, weights, i):
        weights = self.active_fit_one(x, x_pas, y, weights, i)
        weights = self.passive_fit_one(x, weights, i)
        return weights
    def run_scheme(self, active_cond: Callable, X_train, y_train, X_val, y_val, num_it: int = 5000):
        """
        Run the whole training scheme by packaging up training data and
        executing the update loop.
        :param active_cond: Function specifying if iteration i is an active or only passive trial.
        :param X_train: Array with active training samples.
        :param y_train: Array of active training labels.
        :param X_val: Array with active validation samples.
        :param y_val: Array of active validation labels.
        :param num_it: Number of total (active + passive) iterations.
        """
        self.X_val, self.y_val = X_val, y_val
        self.num_it = num_it
        active_bool = active_cond(jnp.arange(num_it), num_it)
        print(active_bool.shape)
        data_train = self.data_prep(X_train, y_train, active_bool, num_it)
        weights = self.weights
        #Training loop
        self.metrics = jax.lax.scan(self.update, weights, data_train)

    def psy_curve(self, weights):
        psy_results = {}
        for thresh in self.psy_data.keys():
            output = self.out(self.psy_data[thresh], weights)
            assert len(output) == self.pars.num_val_points
            psy_results[thresh] = jnp.round(output).mean()
        return psy_results
    
    def data_prep(self, X, y, active_bool, num_it):
        """
        Package data into one array for training.
        Note: Since it doesn't matter here, this is not memory-efficient;
           we keep things simple at the expense of some GPU RAM.
        """
        self.key, subkey = jax.random.split(self.key)
        shuf = jax.random.randint(subkey, (num_it,), 0, len(y))
        shuf_psy = jax.random.randint(subkey, (num_it,), 0, len(self.psy_data_train))
        self.num_active_it = active_bool.sum()
        return jnp.c_[X[shuf], self.psy_data_train[shuf_psy], y[shuf], active_bool, jnp.cumsum(active_bool)]

class OneLayer(BaseModel):
    """
    One Layer Neural Network with one readout weight v.
    """
    def __init__(self, key, init_v: jnp.ndarray = None, pars: parclass_v = parclass_v()):
        """
        Constructor.
        Params:
        key: PRNG key
        init_v (Optional): Initial weights. Either init_v or dim need to be given.
        pars (Optional): Learning and data parameters
        """
        self.pars = pars
        self.key = key
        self.dim = pars.dim
        if init_v is not None:
            assert len(init_v) == self.dim
            self.v = init_v
        else:
            key, subkey = jax.random.split(key)
            self.v = jax.random.normal(subkey, shape=(self.dim,))
            self.v = self.v / jnp.linalg.norm(self.v)
        self.weights = self.v
        self.acc_log, self.psy_log = [], []
        self.loss_v = lambda v, x, y: loss(y, single_out(x, v), v, self.pars.lam_sgd_v)
        self.grad = jax.grad(self.loss_v)
        self.key, subkey = jax.random.split(self.key)
        self.psy_data = self.gen_psy_data(subkey)
        self.key, subkey = jax.random.split(self.key)
        self.psy_data_train = self.gen_psy_data(subkey, "train")
        

    def active_fit_one(self, x: jnp.ndarray, x_pas: jnp.ndarray, y, weights, i):
        """
        Fit one point with a supervised learning rule.
        x: Input
        y: label (0 or 1)
        """
        v = weights
        if self.pars.sup_v:
            grd = self.grad(v, x, y)
            v -= self.pars.lr_sgd_v * grd
        return v
            

    def passive_fit_one(self, x, weights, i):
        v = weights
        delta_hebb = hebb_update(v, x, self.out(x, v), self.pars.lr_hebb_v, self.pars.lam_hebb_v)
        v += delta_hebb
        return v

    def passive_fit_one_psy(self, x, x_pas, y, weights, i):
        return self.passive_fit_one(x_pas, weights, i)
        
    def keep_log(self, X_val, y_val, weights):
        """
        Log what needs to be logged.
        """
        if self.pars.debug:
            return (jnp.round(self.out(X_val, weights)).flatten() == y_val).mean(), self.psy_curve(weights), weights
        else:
            return (jnp.round(self.out(X_val, weights)).flatten() == y_val).mean(), self.psy_curve(weights)
        
    def out(self, x: jnp.ndarray, weights):
        """
        Model output.
        Parameters:
        x: Input"""
        v = weights
        return single_out(x, v)

    def gen_psy_data(self, key, toggle="val"):
        """
        Generate test data for psychometric curves.
        """
        if toggle == "val":
            psy_data_X = {}
            means = self.pars.means
            sig = jnp.array([self.pars.sigs[0]])
            for frac in np.linspace(0, 1, num=self.pars.num_psy_points):
                mean = jnp.array([means[0] + frac * (means[1] - means[0])])
                psy_data_X[frac], _, _ = gen_data_x(key, self.pars.num_val_points, self.pars.dim, 1, sig, mean, vec=True)
            return psy_data_X
        elif toggle == "train":
            means = self.pars.means
            fracs = np.linspace(0, 1, num=6)[:, None]
            sig = jnp.array([self.pars.sigs[0]] * len(fracs))
            mean = means[0][None,:] + fracs * (means[1][None,:] - means[0][None,:])
            return gen_data_x(key, self.pars.num_val_points, self.pars.dim, len(fracs), sig, mean, vec=True)[0]


class TwoLayer(BaseModel):
    """
    Two Layer Network with weight W mapping input to hidden layer,
    lateral connections M and readout weight v.
    """
    def __init__(self, key, init_v: jnp.ndarray = None, init_w: jnp.ndarray = None,
                 pars: parclass_wv = parclass_wv()):
        """
        Constructor.
        Params:
        key: PRNG key
        init_v (Optional): Initial weights v.
        init_w (Optional): Initial weights W.        
        pars (Optional): Learning and data parameters.
        """
        self.pars = pars
        self.key = key
        self.dim = pars.dim
        self.dim_hid = pars.dim_hid
        if init_v is not None:
            assert len(init_v) == self.dim_hid
            self.v = init_v
        else:
            key, subkey = jax.random.split(key)
            self.v = jax.random.normal(subkey, shape=(self.dim_hid,))
            self.v = self.v / jnp.linalg.norm(self.v)
        if init_w is not None:
            assert init_w.shape == (self.dim_hid, self.dim)
            self.w = init_w
        else:
            key, subkey = jax.random.split(key)
            self.w = jax.random.normal(key, shape=(self.dim_hid, self.dim))
            self.w = self.w / jnp.linalg.norm(self.w)
        self.minv = jnp.eye(self.dim_hid)
        self.acc_log, self.psy_log = [], []
        self.weights = self.v, self.minv, self.w
        
        self.loss_v = lambda v, w, minv, x, y: sigmoid_cross_entropy_with_logits(
            double_out_logit(x, v, minv, w), y, v, self.pars.lam_sgd_v)
        self.grad_v = jax.grad(self.loss_v)
        self.loss_w = lambda w, v, minv, x, y: sigmoid_cross_entropy_with_logits(
            double_out_logit(x, v, minv, w), y, w, self.pars.lam_sgd_w)
        self.grad_w = jax.grad(self.loss_w)
        self.key, subkey = jax.random.split(self.key)
        self.psy_data = self.gen_psy_data(subkey)
        self.key, subkey = jax.random.split(self.key)
        self.psy_data_train = self.gen_psy_data(subkey, "train")

    def active_fit_one(self, x, x_pas, y, weights, i):
        weights = self.active_fit_one_v(x, y, weights, i)
        weights = self.active_fit_one_w(x, y, weights, i)
        return weights

    def passive_fit_one(self, x, weights, i):
        weights = self.passive_fit_one_v(x, weights, i)
        weights = self.passive_fit_one_w(x, weights, i)
        return weights

    def passive_fit_one_psy(self, x, x_pas, y, weights, i):
        return self.passive_fit_one(x_pas, weights, i)

    def active_fit_one_v(self, x, y, weights, i):
        """
        Fit one point with a supervised learning rule for v.
        :param x: Input
        :param y: label (0 or 1)
        :param weights: weights
        """
        v, minv, w = weights
        if self.pars.sup_v:
            grd = self.grad_v(v, w, minv, x, y)
            eps = self.pars.lr_sgd_v * jnp.exp(-self.pars.lr_sgd_v_decay*i/self.num_active_it)
            v -= eps * grd
        return v, minv, w

    def active_fit_one_w(self, x, y, weights, i):
        """
        Fit one point with a supervised learning rule for w.
        :param x: Input
        :param y: label (0 or 1)
        :param weights: weights
        """
        v, minv, w = weights
        if self.pars.sup_w:
            grd = self.grad_w(w, v, minv, x, y)
            w -= self.pars.lr_sgd_w * grd
        return v, minv, w

    def passive_fit_one_v(self, x, weights, i):
        """
        Fit one point with an unsupervised learning rule for v.
        :param x: Input
        :param weights: weights
        """
        v, minv, w = weights
        if self.pars.unsup_v:
            delta_hebb = hebb_update(v, x @ w.T @ minv.T, self.out(x, weights), self.pars.lr_hebb_v, self.pars.lam_hebb_v)
            v += delta_hebb
        return v, minv, w

    def passive_fit_one_w(self, x, weights, i):
        """
        Fit one point with an unsupervised learning rule for w.
        :param x: Input
        :param weights: weights
        """
        v, minv, w = weights
        if self.pars.hebb_w:
            w += hebb_update(w, x, x@w.T, lam=1, eps=self.pars.lr_fsm_w, vec=True)
        elif self.pars.unsup_w:
            minv, w = fsm_step(x, minv, w, self.pars.lam_fsm_w, self.pars.lr_fsm_w)
        return v, minv, w
            
        
    def keep_log(self, X_val, y_val, weights):
        """
        Log what needs to be logged.
        """
        v, minv, w = weights
        if self.pars.debug:
            return (jnp.round(self.out(X_val, weights)) == y_val).mean(), self.psy_curve(weights), X_val @ w.T @ minv.T, y_val, v
        else:
            return (jnp.round(self.out(X_val, weights)) == y_val).mean(), self.psy_curve(weights), jnp.linalg.norm(w), jnp.linalg.norm(minv)
    
    def out(self, x: jnp.ndarray, weights):
        """
        Model output.
        Parameters:
        :param x: Input
        :param weights: weights
        """
        v, minv, w = weights
        return double_out(x, v, minv, w)
    
    def gen_psy_data(self, key, toggle="val"):
        """
        Generate test data for psychometric curves.
        """
        means = self.pars.means
        if toggle == "val":
            #Generate input samples for validation
            psy_data_X = {}
            sig = jnp.array([self.pars.sigs[0]])
            for frac in np.linspace(0, 1, num=self.pars.num_psy_points):
                mean = jnp.array([means[0] + frac * (means[1] - means[0])])
                psy_data_X[frac], _, _ = gen_data_x(key, self.pars.num_val_points, self.pars.dim, 1, sig, mean, vec=True)
            return psy_data_X
        elif toggle == "train":
            #Generate input samples to be used for passive training
            fracs = np.linspace(0, 1, num=6)[:, None]
            sig = jnp.array([self.pars.sigs[0]] * len(fracs))
            mean = means[0][None,:] + fracs * (means[1][None,:] - means[0][None,:])
            return gen_data_x(key, self.pars.num_val_points, self.pars.dim, len(fracs), sig, mean, vec=True)[0]
            
