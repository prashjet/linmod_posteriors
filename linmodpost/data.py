import numpy as np
from scipy import stats

class Data:

    def __init__(self,
                 lmd=None,
                 y=None,
                 sig=None,
                 cov=None):
        n = lmd.size
        if lmd.size!=y.size:
            errmsg = 'lmd and y must have same length'
            raise ValueError(errmsg)
        self.n = n
        self.lmd = lmd
        self.y = y
        # check sig/cov set correctly
        c1 = sig is None
        c2 = cov is None
        if (c1 and c2) or (not c1 and not c2):
            errmsg = 'One and only one of sig and cov must be set'
            raise ValueError(errmsg)
        if sig is not None:
            self.sig = sig
            self.cov = sig**2. * np.identity(self.n)
        else:
            if cov.shape != (self.n, self.n):
                errmsg = 'Data covariance must be (n,n) matrix'
                raise ValueError(errmsg)
            self.cov = cov

class MockData(Data):

    def __init__(self,
                 model_grid=None,
                 df=None,
                 sig=None,
                 cov=None):
        # check data parmeters set correctly
        c1 = sig is None
        c2 = cov is None
        if (c1 and c2) or (not c1 and not c2):
            errmsg = 'One and only one of sig and cov must be set'
            raise ValueError(errmsg)
        if sig is not None:
            cov00 = sig**2. * np.identity(model_grid.n)
        else:
            if cov.shape != (model_grid.n, model_grid.n):
                errmsg = 'Data covariance must be (n,n) matrix'
                raise ValueError(errmsg)
            cov00 = cov
        self.ybar = np.dot(model_grid.X, df.beta)
        if sig is not None:
            eps = np.random.normal(loc=0, scale=sig, size=model_grid.n)
        else:
            mvn = stats.multivariate_normal(mean=np.zeros(model_grid.n),
                                            cov=cov00)
            eps = mvn.rvs()
        y = self.ybar + eps
        if sig is not None:
            super().__init__(lmd=model_grid.lmd,
                             y=y,
                             sig=sig)
        else:
            super().__init__(lmd=model_grid.lmd,
                             y=y,
                             cov=cov)
