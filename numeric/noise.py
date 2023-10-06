"""
Functions to add additive noise to given dataset
"""
import numpy as np


class Noise:
    """
    Add Noise to a given dataset
    """

    def __init__(self, *args):
        self.name = "noise_object"

    def __call__(self, x: np.ndarray, axis: None | int = None) -> np.ndarray:
        """
        Signature of noise adding function

        Parameters
        ----------
        x : np.ndarray
            dataset with shape (n, n_states)
        axis : int | None
            optional axis to add noise to all are used if not given
        Returns
        -------
        np.ndarray
            noisy dataset with same shape
        """
        raise NotImplementedError


class NoNoise(Noise):

    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray, axis=None) -> np.ndarray:
        """
        Just don't do anything...

        Parameters
        ----------
        x : np.ndarray
            vector to ignore
        axis : int
            ignored

        Returns
        -------
        np.ndarray
            just input x
        """
        return x

    def __str__(self):
        description = f"No Noise added"

        return description

    def __repr__(self):
        description = f"NoNoise()"

        return description


class Gauss(Noise):

    def __init__(self, R: float | np.ndarray, mean: float | np.ndarray = 0, axis=None,
                 outlier_params: tuple | None = None):
        """
        Create a gaussian noise add-function with given covariance and optional outliers

        Parameters
        ----------
        R : float | np.ndarray
            noise variance can be a covariance matrix or a vector/scalar from which a diagonal matrix will be
            constructed
        mean : float | np.ndarray
            mean of noise, 0 by default. can be float or array and will be constructed as R
        axis : int
            state to add noise to, all will be used if not given
        outlier_params : tuple or None
            parameters for outliers (outlier_ratio, index_outliers) with types (float, int)
        """
        # check if outlier ratio is reasonable
        if outlier_params is not None and not 0 <= outlier_params[0] <= 1:
            raise ValueError("Outlier ratio must be between 0 and 1")

        super().__init__()

        self.name = "gaussian noise"

        # set noise parameters
        self.R = R
        self.mean = mean
        self.outlier_params = outlier_params
        self.axis = axis

    def __call__(self, x: np.ndarray, axis=None) -> np.ndarray:
        """
        Add gaussian noise to a dataset x

        Parameters
        ----------
        x : np.ndarray
            dataset with shape (n, n_states)
        axis : int
            optional axis to add noise to, all axis are used if not given

        Returns
        -------
        np.ndarray
            noise dataset with shape (n, n_states)
        """
        axis = self.axis if axis is None else axis

        # check if axis is out or range f given
        if axis is not None and axis >= x.shape[1]:
            raise AttributeError(f"Axis is out of range or states {x.shape[1]}")

        # make matrix if R is scalar or vector
        if isinstance(self.R, (float, int)) or self.R.shape[0] == 1:
            R = np.ones((x.shape[-1])) * self.R
        elif self.R.ndim == 1 and self.R.shape[0] == x.shape[1]:
            R = np.diag(self.R)
        elif self.R.ndim == 2 and self.R.shape[0] == self.R.shape[1] == x.shape[1]:
            R = self.R
        else:
            raise AttributeError("R must be scalar or vector/matrix fitting axis 1 of x")

        # make matrix if mean is scalar or vector
        if isinstance(self.mean, (float, int)) or self.mean.shape[0] == 1:
            mean = np.ones((x.shape[-1])) * self.mean
        elif self.mean.ndim == 1 and self.mean.shape[0] == x.shape[1]:
            mean = np.diag(self.mean)
        elif self.mean.ndim == 2 and self.mean.shape[0] == self.mean.shape[1] == x.shape[1]:
            mean = self.mean
        else:
            raise AttributeError("R must be scalar or vector/matrix fitting axis 1 of x")

        # add noise
        if axis is None:
            x_noise = x + np.random.normal(loc=mean, scale=R, size=x.shape)
        else:
            x[:, axis] = x[:, axis] + np.random.normal(loc=mean, scale=R[axis], size=x[:, axis].shape)
            x_noise = x

        if self.outlier_params is not None:
            x_noise = self._add_outliers(x_noise)

        return x_noise

    def _add_outliers(self, x: np.ndarray) -> np.ndarray:
        """
        Add outliers with saved ratio to saved outlier axis (like a scale)

        Parameters
        ----------
        x : np.ndarray
            dataset with shape (n, n_states)

        Returns
        -------
        np.ndarray
            dataset with added outliers with shape (n, n_states)
        """
        # unpack arguments
        outlier_ratio, index_outlier = self.outlier_params

        if index_outlier >= x.shape[1]:
            raise AttributeError(f"Index of outliers is out or range for number of states: "
                                 f"{x.shape[1]}<={index_outlier}")

        # get length of dataset and number of outliers
        length_data = len(x[:, index_outlier])
        n_outliers = int(length_data * outlier_ratio)

        # randomly select indices for outliers
        indices = np.random.choice(np.arange(length_data), n_outliers)

        # actually add outliers
        steps = np.random.choice(a=[-1, 1], size=length_data)
        x[indices, index_outlier] = x[indices, index_outlier] + steps[indices + 1] * np.random.uniform(
            low=1, high=10, size=(len(indices),))

        return x

    def __str__(self):
        description = (f"Gaussian noise generator\n"
                       f"  - R: {self.R}\n"
                       f"  - axis: {self.axis}\n"
                       f"  - outlier ratio: {self.outlier_params[0] if self.outlier_params is not None else None}\n"
                       f"  - outlier axis: {self.outlier_params[1] if self.outlier_params is not None else None}")

        return description

    def __repr__(self):
        representation = f"Gauss(R={self.R}, axis={self.axis}, outlier_params={self.outlier_params})"

        return representation


class PinkNoise(Noise):
    pass


class BrownianNoise(Noise):
    pass
