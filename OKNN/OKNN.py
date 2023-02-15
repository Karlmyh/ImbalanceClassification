

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier, KernelDensity
from sklearn.metrics import recall_score

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
        n_neighbors_density = None,
        h = None,
        n_neighbors = 10,
        random_state = 1,
    ):
        self.n_neighbors_density = n_neighbors_density
        self.h = h
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
    def oversample_knn(self, X, n_sample):
        
        
        
        sampler = UnitBallSampler(self.dim, self.random_state)
        over_samples = sampler.sample(n_sample )
        
        kdtree = KDTree(X)
        
        np.random.seed(self.random_state)
        neighbors = X[np.random.choice(X.shape[0], n_sample )]
        k_dist, _ = kdtree.query(neighbors, k = self.n_neighbors_density + 1)
        k_dist = k_dist[:,-1]
        
        X_oversample = neighbors + (over_samples.T * k_dist).T
        
        return X_oversample
        

    def oversample_kde(self, X, n_sample):
        
        
        
        sampler = KernelDensity(bandwidth = self.h).fit(X)
        X_oversample = sampler.sample(n_sample)
        
        return X_oversample
        
        
        
    def fit(self, X, y):
        
        self.n_sample, self.dim = X.shape
        
        if self.h is not None:
            self.oversample = self.oversample_kde
        elif self.n_neighbors_density is not None:
            self.oversample = self.oversample_knn
            
            
        labels = np.unique(y)
        n_class = labels.shape[0]
        n_samples = [(y==label).sum() for label in labels]
        major_class = np.argmax(n_samples)
        n_major = np.max(n_samples)
        
        
            
        X_whole = X[y == major_class]
        y_whole = np.repeat(major_class, n_major)
        
        for label in labels:
            
            if label == major_class:
                continue
            else:
                X_whole = np.vstack([X_whole, self.oversample(X[y==label], n_major)])
                y_whole = np.append(y_whole, np.repeat(label, n_major))
        
        
        self.X_whole = X_whole
        self.y_whole = y_whole
        
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
        return - recall_score( y_true = y,
                               y_pred = self.predict(X),
                               average = "macro") 
        
        
                
                
            
                
        
        
        
    
    

    

