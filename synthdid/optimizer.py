from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import partial
from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from bayes_opt import BayesianOptimization


class Optimize(object):
    ####
    # Synthetic Difference in Differences (SDID)
    ####
    def est_zeta(self, Y_pre_c) -> float:
        """
        # SDID
        Parameter to adjust the L2 penalty term
        """
        return (self.n_treat * self.n_post_term) ** (1 / 4) * np.std(
            Y_pre_c.diff().dropna().values
        )

    def est_omega(self, Y_pre_c, Y_pre_t, zeta):
        """
        # SDID
        estimating omega
        """
        Y_pre_t = Y_pre_t.copy()
        n_features = Y_pre_c.shape[1]
        nrow = Y_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)
        _w0 = 1

        start_w = np.append(_w, _w0)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

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

    def est_lambda(self, Y_pre_c, Y_post_c):
        """
        # SDID
        estimating lambda
        """
        Y_pre_c_T = Y_pre_c.T
        Y_post_c_T = Y_post_c.T

        n_pre_term = Y_pre_c_T.shape[1]

        _lambda = np.repeat(1 / n_pre_term, n_pre_term)
        _lambda0 = 1

        start_lambda = np.append(_lambda, _lambda0)

        if type(Y_post_c_T) == pd.core.frame.DataFrame:
            Y_post_c_T = Y_post_c_T.mean(axis=1)

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

    def l2_loss(self, W, X, y, zeta, nrow) -> float:
        """
        Loss function with L2 penalty
        """
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        _X["intersept"] = 1
        return np.sum((y - _X.dot(W)) ** 2) + nrow * zeta ** 2 * np.sum(W[:-1] ** 2)

    ####
    # Synthetic Control Method (SC)
    ####
    def rmse_loss(self, W, X, y, intersept=True) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        if intersept:
            _X["intersept"] = 1
        return np.mean(np.sqrt((y - _X.dot(W)) ** 2))

    def rmse_loss_with_V(self, W, V, X, y) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _rss = (y - X.dot(W)) ** 2

        _n = len(y)
        _importance = np.zeros((_n, _n))

        np.fill_diagonal(_importance, V)

        return np.sum(_importance @ _rss)

    def _v_loss(self, V, X, y, return_loss=True):
        Y_pre_t = self.Y_pre_t.copy()

        n_features = self.Y_pre_c.shape[1]
        _w = np.repeat(1 / n_features, n_features)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        w_bnds = tuple((0, 1) for i in range(n_features))
        _caled_w = fmin_slsqp(
            partial(self.rmse_loss_with_V, V=V, X=X, y=y),
            _w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=w_bnds,
            disp=False,
        )
        if return_loss:
            return self.rmse_loss(_caled_w, self.Y_pre_c, Y_pre_t, intersept=False)
        else:
            return _caled_w

    def estimate_v(self, additional_X, additional_y):
        _len = len(additional_X)
        _v = np.repeat(1 / _len, _len)

        caled_v = fmin_slsqp(
            partial(self._v_loss, X=additional_X, y=additional_y),
            _v,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=tuple((0, 1) for i in range(_len)),
            disp=False,
        )
        return caled_v

    def est_omega_ADH(
        self, Y_pre_c, Y_pre_t, additional_X=pd.DataFrame(), additional_y=pd.DataFrame()
    ):
        """
        # SC
        estimating omega for synthetic control method (not for synthetic diff.-in-diff.)
        """
        Y_pre_t = Y_pre_t.copy()

        n_features = Y_pre_c.shape[1]
        nrow = Y_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        # Required to have non negative values
        w_bnds = tuple((0, 1) for i in range(n_features))

        if len(additional_X) == 0:
            caled_w = fmin_slsqp(
                partial(self.rmse_loss, X=Y_pre_c, y=Y_pre_t, intersept=False),
                _w,
                f_eqcons=lambda x: np.sum(x) - 1,
                bounds=w_bnds,
                disp=False,
            )

            return caled_w
        else:
            assert additional_X.shape[1] == Y_pre_c.shape[1]
            if type(additional_y) == pd.core.frame.DataFrame:
                additional_y = additional_y.mean(axis=1)

            # normalized
            temp_df = pd.concat([additional_X, additional_y], axis=1)
            ss = StandardScaler()
            ss_df = pd.DataFrame(
                ss.fit_transform(temp_df), columns=temp_df.columns, index=temp_df.index
            )

            ss_X = ss_df.iloc[:, :-1]
            ss_y = ss_df.iloc[:, -1]

            add_X = pd.concat([Y_pre_c, ss_X])
            add_y = pd.concat([Y_pre_t, ss_y])

            self.caled_v = self.estimate_v(additional_X=add_X, additional_y=add_y)

            return self._v_loss(self.caled_v, X=add_X, y=add_y, return_loss=False)

    #####
    # cv search for zeta
    ####

    def _zeta_given_cv_loss_inverse(self, zeta, cv=5, split_type="KFold"):
        return -1 * self._zeta_given_cv_loss(zeta, cv, split_type)[0]

    def _zeta_given_cv_loss(self, zeta, cv=5, split_type="KFold"):
        nrow = self.Y_pre_c.shape[0]
        if split_type == "KFold":
            kf = KFold(n_splits=cv, random_state=self.random_seed)
        elif split_type == "TimeSeriesSplit":
            kf = TimeSeriesSplit(n_splits=cv)
        elif split_type == "RepeatedKFold":
            _cv = max(2, int(cv / 2))
            kf = RepeatedKFold(
                n_splits=_cv, n_repeats=_cv, random_state=self.random_seed
            )

        loss_result = []
        nf_result = []
        for train_index, test_index in kf.split(self.Y_pre_c, self.Y_pre_t):
            train_w = self.est_omega(
                self.Y_pre_c.iloc[train_index], self.Y_pre_t.iloc[train_index], zeta
            )

            nf_result.append(np.sum(np.round(np.abs(train_w), 3) > 0) - 1)

            loss_result.append(
                self.rmse_loss(
                    train_w,
                    self.Y_pre_c.iloc[test_index],
                    self.Y_pre_t.iloc[test_index],
                )
            )
        return np.mean(loss_result), np.mean(nf_result)

    def grid_search_zeta(
        self, cv=5, n_candidate=20, candidate_zata=[], split_type="KFold"
    ):
        """
        Search for zeta using grid search instead of theoretical values
        """

        if len(candidate_zata) == 0:

            for _z in np.linspace(0.1, self.base_zeta * 2, n_candidate):
                candidate_zata.append(_z)

            candidate_zata.append(self.base_zeta)
            candidate_zata.append(0)

            candidate_zata = sorted(candidate_zata)

            result_loss_dict = {}
            result_nf_dict = {}

        print("cv: zeta")
        for _zeta in tqdm(candidate_zata):
            result_loss_dict[_zeta], result_nf_dict[_zeta] = self._zeta_given_cv_loss(
                _zeta, cv=cv, split_type=split_type
            )

        loss_sorted = sorted(result_loss_dict.items(), key=lambda x: x[1])

        return loss_sorted[0]

    def bayes_opt_zeta(
        self,
        cv=5,
        init_points=5,
        n_iter=5,
        zeta_max=None,
        zeta_min=None,
        split_type="KFold",
    ):
        """
        Search for zeta using Bayesian Optimization instead of theoretical values
        """
        if zeta_max == None:
            zeta_max = self.base_zeta * 1.02
            zeta_max2 = self.base_zeta * 2

        if zeta_min == None:
            zeta_min = self.base_zeta * 0.98
            zeta_min2 = 0.01

        pbounds = {"zeta": (zeta_min, zeta_max)}

        optimizer = BayesianOptimization(
            f=partial(self._zeta_given_cv_loss_inverse, cv=cv, split_type=split_type),
            pbounds=pbounds,
            random_state=self.random_seed,
        )

        optimizer.maximize(
            init_points=2,
            n_iter=2,
        )

        optimizer.set_bounds(new_bounds={"zeta": (zeta_min2, zeta_max2)})

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        optimizer.max["params"]["zeta"]

        return (optimizer.max["params"]["zeta"], optimizer.max["target"] * -1)

    #####
    # The following is for sparse estimation
    ####
    def est_omega_ElasticNet(self, Y_pre_c, Y_pre_t):
        Y_pre_t = Y_pre_t.copy()

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)
            # Y_pre_t.columns = "treatment_group"

        regr = ElasticNetCV(cv=5, random_state=0)
        regr.fit(Y_pre_c, Y_pre_t)

        self.elastic_net_alpha = regr.alpha_

        caled_w = regr.coef_

        return np.append(caled_w, regr.intercept_)

    def est_omega_Lasso(self, Y_pre_c, Y_pre_t):
        Y_pre_t = Y_pre_t.copy()

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        regr = LassoCV(cv=5, random_state=0)
        regr.fit(Y_pre_c, Y_pre_t)

        self.lasso_alpha = regr.alpha_

        caled_w = regr.coef_

        return np.append(caled_w, regr.intercept_)

    def est_omega_Ridge(self, Y_pre_c, Y_pre_t):
        Y_pre_t = Y_pre_t.copy()

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        regr = RidgeCV(cv=5)
        regr.fit(Y_pre_c, Y_pre_t)

        self.ridge_alpha = regr.alpha_

        caled_w = regr.coef_

        return np.append(caled_w, regr.intercept_)
