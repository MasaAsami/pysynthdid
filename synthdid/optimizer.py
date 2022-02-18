from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import reduce, partial
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization


class Optimize(object):
    def est_zeta(self):
        return (self.n_treat * self.n_post_term) ** (1 / 4) * self.Y_pre_c.std().std()

    def l2_loss(self, W, X, y, zeta, nrow) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        _X["intersept"] = 1
        return np.sum((y - _X.dot(W)) ** 2) + nrow * zeta ** 2 * np.sum(W ** 2)

    def mse_loss(self, W, X, y, intersept=True) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        if intersept:
            _X["intersept"] = 1
        return np.mean(np.sqrt((y - _X.dot(W)) ** 2))

    def est_omega(self, Y_pre_c, Y_pre_t, zeta):
        Y_pre_t = Y_pre_t.copy()
        n_features = Y_pre_c.shape[1]
        nrow = Y_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)
        _w0 = 1

        start_w = np.append(_w, _w0)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)
            # Y_pre_t.columns = "treatment_group"

        # Required to have non negative values
        max_bnd = abs(Y_pre_t.mean()) * 2
        w_bnds = tuple(
            (0, 1) if i < n_features else (max_bnd * -1, max_bnd)
            for i in range(n_features + 1)
        )

        caled_w = fmin_slsqp(
            partial(self.l2_loss, X=Y_pre_c, y=Y_pre_t, zeta=zeta, nrow=nrow),
            start_w,
            f_eqcons=lambda x: np.sum(x[:n_features]) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w

    def est_omega_ADH(self):
        Y_pre_t = self.Y_pre_t.copy()

        n_features = self.Y_pre_c.shape[1]
        nrow = self.Y_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)
            # Y_pre_t.columns = "treatment_group"

        # normalized
        # temp_df = pd.concat([Y_pre_c, Y_pre_t], axis=1)
        # ss = StandardScaler()
        # ss_df = pd.DataFrame(ss.fit_transform(temp_df) , columns=temp_df.columns, index=temp_df.index)

        # ss_Y_pre_c = ss_df.iloc[:,:-1]
        # ss_Y_pre_t = ss_df.iloc[:,-1]

        # Required to have non negative values
        w_bnds = tuple((0, 1) for i in range(n_features))

        caled_w = fmin_slsqp(
            partial(self.mse_loss, X=self.Y_pre_c, y=Y_pre_t, intersept=False),
            _w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w

    def est_lambda(self):
        Y_pre_c_T = self.Y_pre_c.T
        Y_post_c_T = self.Y_post_c.T

        n_pre_term = Y_pre_c_T.shape[1]

        _lambda = np.repeat(1 / n_pre_term, n_pre_term)
        _lambda0 = 1

        start_lambda = np.append(_lambda, _lambda0)

        if type(Y_post_c_T) == pd.core.frame.DataFrame:
            Y_post_c_T = Y_post_c_T.mean(axis=1)
            # Y_post_c_T.columns = "mean_Y_post"

        # Required to have non negative values
        max_bnd = abs(Y_post_c_T.mean()) * 2
        lambda_bnds = tuple(
            (0, 1) if i < n_pre_term else (max_bnd * -1, max_bnd)
            for i in range(n_pre_term + 1)
        )

        caled_lambda = fmin_slsqp(
            partial(self.l2_loss, X=Y_pre_c_T, y=Y_post_c_T, zeta=0, nrow=0),
            start_lambda,
            f_eqcons=lambda x: np.sum(x[:n_pre_term]) - 1,
            bounds=lambda_bnds,
            disp=False,
        )

        return caled_lambda[:n_pre_term]

    def _zeta_given_cv_loss_inverse(self, zeta, cv=5):
        return -1 * self._zeta_given_cv_loss(zeta, cv)[0]

    def _zeta_given_cv_loss(self, zeta, cv=5):
        nrow = self.Y_pre_c.shape[0]
        kf = KFold(n_splits=cv)
        loss_result = []
        nf_result = []
        for train_index, test_index in kf.split(self.Y_pre_c, self.Y_pre_t):
            train_w = self.est_omega(
                self.Y_pre_c.iloc[train_index], self.Y_pre_t.iloc[train_index], zeta
            )

            nf_result.append(np.sum(np.round(np.abs(train_w), 3) > 0) - 1)

            loss_result.append(
                self.mse_loss(
                    train_w,
                    self.Y_pre_c.iloc[test_index],
                    self.Y_pre_t.iloc[test_index],
                )
            )
        return np.mean(loss_result), np.mean(nf_result)

    def gread_search_zeta(self, cv=5, candidate_zata=[]):

        if len(candidate_zata) == 0:

            for _z in np.linspace(0.1, self.base_zeta * 2, 30):
                candidate_zata.append(_z)

            candidate_zata.append(self.base_zeta)
            candidate_zata.append(0)

            candidate_zata = sorted(candidate_zata)

            result_loss_dict = {}
            result_nf_dict = {}

        print("cv: zeta")
        for _zeta in tqdm(candidate_zata):
            result_loss_dict[_zeta], result_nf_dict[_zeta] = self._zeta_given_cv_loss(
                _zeta
            )

        loss_sorted = sorted(result_loss_dict.items(), key=lambda x: x[1])

        return loss_sorted[0]

    def bayes_opt_zeta(
        self,
        cv=5,
        init_points=5,
        n_iter=10,
        zeta_max=None,
        zeta_min=None,
    ):
        if zeta_max == None:
            zeta_max = self.base_zeta * 2

        if zeta_min == None:
            zeta_min = 0.01

        pbounds = {"zeta": (zeta_min, zeta_max)}

        optimizer = BayesianOptimization(
            f=partial(self._zeta_given_cv_loss_inverse, cv=cv),
            pbounds=pbounds,
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        optimizer.max["params"]["zeta"]

        return (optimizer.max["params"]["zeta"], optimizer.max["target"] * -1)
