"""
Adaptive Weighted Nearest Neighbor Density Estimation
-----------------------------------------------------
"""

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.metrics import mean_squared_error

class UnitBallSampler(object):
    def __init__(
            self,
            dim,
            random_state = 1,
            ):
    
        self.dim = dim
        self.random_state = random_state
        
    def sample(self, n_sample):
        np.random.seed(self.random_state)
        
        sample = np.random.normal(size = self.dim * n_sample).reshape(-1, self.dim)
        sample = sample.T / np.linalg.norm(sample, axis = 1) * np.random.rand(n_sample)**(1/self.dim)
        
        return sample.T




class OKNN(object):
    
    def __init__(
        self,
        *,
        n_neighbors_density = 5,
        n_neighbors = 10,
        random_state = 1,
    ):
        self.n_neighbors_density = n_neighbors_density
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        

    def fit(self, X, y):
        

        
        kdtree = KDTree(X)
        self.n_sample, self.dim = X.shape

        X_minor = X[ y == 1, :]
        n_minor = X_minor.shape[0]
        X_major = X[ y == 0, :]
        n_major = X_major.shape[0]
        
        sampler = UnitBallSampler(self.dim, self.random_state)
        over_samples = sampler.sample(n_major - n_minor)
        
        np.random.seed(self.random_state)
        neighbors = X_minor[np.random.choice(n_minor, n_major - n_minor)]
        k_dist, _ = kdtree.query(neighbors, k = self.n_neighbors_density + 1)
        k_dist = k_dist[:,-1]
        
        X_oversample = neighbors + (over_samples.T * k_dist).T
        
        X_whole = np.vstack([X, X_oversample])
        y_whole = np.append(y, np.ones(n_major - n_minor))
        
        self.classifier = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        self.classifier.fit(X_whole, y_whole)
        
        
        
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['n_neighbors_density',"n_neighbors"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self



       
    
    def predict(self, X):
        return self.classifier.predict(X)
  

    def score(self, X, y):
        return mean_squared_error(self.predict(X), y)
        
        
                
                
            
                
        
        
        
    
    

    

