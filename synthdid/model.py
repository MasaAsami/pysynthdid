from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import reduce, partial
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import seaborn as sns
import matplotlib.pyplot as plt

from synthdid.optimizer import Optimize

class SynthDID(Optimize):
    """
    hogehoge
    """

    def __init__(
        self, df, pre_term, post_term, treatment_unit: list, random_seed=0, **kwargs
    ):
        # first val
        self.df = df
        self.pre_term = pre_term
        self.post_term = post_term
        self.random_seed = random_seed
        # able to change
        self.treatment = treatment_unit
        self.control = [col for col in df.columns if col not in self.treatment]
        self._divide_data()

        # params
        self.hat_zeta = None
        self.base_zeta = None
        self.hat_omega_ADH = None
        self.hat_omega = None
        self.hat_lambda = None

    def _divide_data(self):

        self.Y_pre_c = self.df.loc[self.pre_term[0] : self.pre_term[1], self.control]
        self.Y_pre_t = self.df.loc[self.pre_term[0] : self.pre_term[1], self.treatment]

        self.Y_post_c = self.df.loc[self.post_term[0] : self.post_term[1], self.control]
        self.Y_post_t = self.df.loc[
            self.post_term[0] : self.post_term[1], self.treatment
        ]

        self.n_treat = len(self.treatment)
        self.n_post_term = len(self.Y_post_t)

    def fit(self, model="all", zeta_type="base"):

        self.base_zeta = self.est_zeta()

        if zeta_type == "base":
            self.zeta = self.base_zeta

        elif zeta_type == "grid_search":
            self.zeta = self.gread_search_zeta(cv=5, candidate_zata=[])[0]

        elif zeta_type == "bayesian_opt":
            self.zeta = self.bayes_opt_zeta(cv=5)[0]

        else:
            print(f"your choice :{zeta_type} is not supported.")
            self.zeta = self.base_zeta

        self.hat_omega = self.est_omega(self.Y_pre_c, self.Y_pre_t, np.round(self.zeta,10))
        self.hat_omega_ADH = self.est_omega_ADH()
        self.hat_lambda = self.est_lambda()

    def did_potentical_outcome(self):
        """
        return potential outcome
        """
        Y_pre_c = self.Y_pre_c.copy()
        Y_pre_t = self.Y_pre_t.copy()
        Y_post_c = self.Y_post_c.copy()
        Y_post_t = self.Y_post_t.copy()

        if type(Y_pre_t) != pd.DataFrame:
            Y_pre_t = pd.DataFrame(Y_pre_t)

        if type(Y_post_t) != pd.DataFrame:
            Y_post_t = pd.DataFrame(Y_post_t)

        Y_pre_t["did"] = Y_pre_t.mean(axis=1).mean()
        Y_post_t["did"] = (
            Y_pre_t.mean(axis=1).mean()
            + Y_post_c.mean(axis=1).mean()
            - Y_pre_c.mean(axis=1).mean()
        )

        return pd.concat([Y_pre_t["did"], Y_post_t["did"]], axis=0)

    def sc_potentical_outcome(self):
        return pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_ADH)

    def sdid_potentical_outcome(self):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1

        base_sc = Y_post_c_intercept @ self.hat_omega
        lambda_effect = (self.Y_pre_t.T @ self.hat_lambda).values[0]
        sc_pretrend_with_timeweighted = (
            Y_pre_c_intercept @ self.hat_omega @ self.hat_lambda
        )

        post_outcome = base_sc + lambda_effect - sc_pretrend_with_timeweighted

        return pd.concat([Y_pre_c_intercept.dot(self.hat_omega), post_outcome], axis=0)

    def target_y(self):
        return self.df.loc[self.pre_term[0] : self.post_term[1], self.treatment].mean(
            axis=1
        )
    
    def estimated_params(self, model="sdid"):
        if model=="sdid":
            Y_pre_c_intercept = self.Y_pre_c.copy()
            Y_post_c_intercept = self.Y_post_c.copy()
            Y_pre_c_intercept["intercept"] = 1
            Y_post_c_intercept["intercept"] = 1
            return (
                pd.DataFrame({"features":Y_pre_c_intercept.columns, "sdid_weight":np.round(self.hat_omega,3) }) ,
                pd.DataFrame({"time":Y_pre_c_intercept.index, "sdid_weight":np.round(self.hat_lambda,3) }) 
            )
        elif model=="sc":
            return pd.DataFrame({"features":self.Y_pre_c.columns, "sc_weight":np.round(self.hat_omega_ADH,3) }) 
        else:
            return None


    def delta_plot(self, model="all"):
        result = pd.DataFrame({"actual_y": self.target_y()})
        
        result["did"] = self.did_potentical_outcome()
        result["sc"] = self.sc_potentical_outcome()
        result["sdid"] = self.sdid_potentical_outcome()

        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        result["actual_y"].plot(ax=ax, color="black", linewidth=1, label="actual_y")
        result["did"].plot(ax=ax, linewidth=1, label="Difference in Differences")
        result["sc"].plot(ax=ax, label="Synthetic Control")
        result["sdid"].plot(ax=ax, label="Synthetic Differences in Differences")

        ax.axvspan(self.post_term[0], self.post_term[1], alpha=0.3, color="lightblue")
        plt.title("")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from sample_data import fetch_CaliforniaSmoking
    df = fetch_CaliforniaSmoking()

    PRE_TEREM = [1970, 1979]
    POST_TEREM = [1980, 1988]
    TREATMENT = ["California"]

    sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
    sdid.base_zeta
    sdid.fit(zeta_type="base")
    sdid.base_zeta
    sdid.zeta
    sdid.fit(zeta_type="grid_search")
    sdid.zeta
    sdid.fit(zeta_type="bayesian_opt")
    sdid.base_zeta
    sdid.did_potentical_outcome()
    sdid.sdid_potentical_outcome()
    sdid.sc_potentical_outcome()
    sdid.did_plot()
    sdid.zeta
    sdid.hat_omega
    sdid.hat_omega_ADH
    sdid.control
    sdid.Y_pre_c
