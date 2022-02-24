from code import interact
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
        self.hat_omega_ElasticNet = None
        self.hat_omega_Lasso = None
        self.hat_omega_Ridge = None

    def _divide_data(self):

        self.Y_pre_c = self.df.loc[self.pre_term[0] : self.pre_term[1], self.control]
        self.Y_pre_t = self.df.loc[self.pre_term[0] : self.pre_term[1], self.treatment]

        self.Y_post_c = self.df.loc[self.post_term[0] : self.post_term[1], self.control]
        self.Y_post_t = self.df.loc[
            self.post_term[0] : self.post_term[1], self.treatment
        ]

        self.n_treat = len(self.treatment)
        self.n_post_term = len(self.Y_post_t)

    def fit(
        self,
        model="all",
        zeta_type="base",
        force_zeta=None,
        sparce_estimation=False,
        cv=5,
        cv_split_type="KFold",
        candidate_zata=[],
        n_candidate=20,
        simple_sc=True,
    ):

        self.base_zeta = self.est_zeta()

        if zeta_type == "base":
            self.zeta = self.base_zeta

        elif zeta_type == "grid_search":
            self.zeta = self.gread_search_zeta(
                cv=cv,
                n_candidate=n_candidate,
                candidate_zata=candidate_zata,
                split_type=cv_split_type,
            )[0]

        elif zeta_type == "bayesian_opt":
            self.zeta = self.bayes_opt_zeta(cv=cv, split_type=cv_split_type)[0]

        else:
            print(f"your choice :{zeta_type} is not supported.")
            self.zeta = self.base_zeta

        if force_zeta != None:
            self.zeta = force_zeta

        self.hat_omega = self.est_omega(self.Y_pre_c, self.Y_pre_t, self.zeta)
        self.hat_omega_ADH = self.est_omega_ADH(simple_sc=simple_sc)
        self.hat_lambda = self.est_lambda()

        if sparce_estimation:
            self.hat_omega_ElasticNet = self.est_omega_ElasticNet(
                self.Y_pre_c, self.Y_pre_t
            )
            self.hat_omega_Lasso = self.est_omega_Lasso(self.Y_pre_c, self.Y_pre_t)
            self.hat_omega_Ridge = self.est_omega_Ridge(self.Y_pre_c, self.Y_pre_t)

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

        Y_pre_t["did"] = Y_pre_t.mean(axis=1)

        base_trend = Y_post_c.mean(axis=1)

        Y_post_t["did"] = base_trend + (
            Y_pre_t.mean(axis=1).mean() - Y_pre_c.mean(axis=1).mean()
        )

        return pd.concat([Y_pre_t["did"], Y_post_t["did"]], axis=0)

    def sc_potentical_outcome(self):
        return pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_ADH)

    def sparceReg_potentical_outcome(self, model="ElasticNet"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1

        if model == "ElasticNet":
            s_omega = self.hat_omega_ElasticNet
        elif model == "Lasso":
            s_omega = self.hat_omega_Lasso
        elif model == "Ridge":
            s_omega = self.hat_omega_Ridge
        else:
            print(f"model={model} is not supported")
            return None
        return pd.concat([Y_pre_c_intercept, Y_post_c_intercept]).dot(s_omega)

    def sdid_trajectory(self):
        hat_omega = self.hat_omega[:-1]
        Y_c = pd.concat([self.Y_pre_c, self.Y_post_c])
        n_features = self.Y_pre_c.shape[1]
        start_w = np.repeat(1 / n_features, n_features)

        _intercept = (start_w - hat_omega) @ self.Y_pre_c.T @ self.hat_lambda

        return Y_c.dot(hat_omega) + _intercept

    def sdid_potentical_outcome(self):
        Y_pre_c = self.Y_pre_c.copy()
        Y_post_c = self.Y_post_c.copy()
        hat_omega = self.hat_omega[:-1]

        base_sc = Y_post_c @ hat_omega
        pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
        pre_control_base = Y_pre_c @ hat_omega @ self.hat_lambda

        pre_outcome = Y_pre_c.dot(hat_omega)

        post_outcome = base_sc + pre_treat_base - pre_control_base

        return pd.concat([pre_outcome, post_outcome], axis=0)

    def sparce_sdid_potentical_outcome(self, model="ElasticNet"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1

        if model == "ElasticNet":
            s_omega = self.hat_omega_ElasticNet
        elif model == "Lasso":
            s_omega = self.hat_omega_Lasso
        elif model == "Ridge":
            s_omega = self.hat_omega_Ridge
        else:
            print(f"model={model} is not supported")
            return None

        base_sc = Y_post_c_intercept @ s_omega
        pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
        pre_control_base = Y_pre_c_intercept @ s_omega @ self.hat_lambda

        post_outcome = base_sc + pre_treat_base - pre_control_base

        return pd.concat([Y_pre_c_intercept.dot(s_omega), post_outcome], axis=0)

    def target_y(self):
        return self.df.loc[self.pre_term[0] : self.post_term[1], self.treatment].mean(
            axis=1
        )

    def estimated_params(self, model="sdid"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1
        if model == "sdid":
            return (
                pd.DataFrame(
                    {
                        "features": Y_pre_c_intercept.columns,
                        "sdid_weight": np.round(self.hat_omega, 3),
                    }
                ),
                pd.DataFrame(
                    {
                        "time": Y_pre_c_intercept.index,
                        "sdid_weight": np.round(self.hat_lambda, 3),
                    }
                ),
            )
        elif model == "sc":
            return pd.DataFrame(
                {
                    "features": self.Y_pre_c.columns,
                    "sc_weight": np.round(self.hat_omega_ADH, 3),
                }
            )
        elif model == "ElasticNet":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "ElasticNet_weight": np.round(self.hat_omega_ElasticNet, 3),
                }
            )
        elif model == "Lasso":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "Lasso_weight": np.round(self.hat_omega_Lasso, 3),
                }
            )
        elif model == "Ridge":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "Ridge_weight": np.round(self.hat_omega_Ridge, 3),
                }
            )
        else:
            return None

    def comparison_plot(self, model="all", figsize=(10, 7)):
        result = pd.DataFrame({"actual_y": self.target_y()})

        result["did"] = self.did_potentical_outcome()
        result["sc"] = self.sc_potentical_outcome()
        result["sdid"] = self.sdid_potentical_outcome()

        result = result.loc[self.post_term[0] : self.post_term[1]]

        fig, ax = plt.subplots(figsize=figsize)

        result["actual_y"].plot(ax=ax, color="black", linewidth=1, label="actual_y")
        result["sdid"].plot(ax=ax, label="Synthetic Difference in Differences")
        result["sc"].plot(
            ax=ax, linewidth=1, linestyle="dashed", label="Synthetic Control"
        )
        result["did"].plot(
            ax=ax, linewidth=1, linestyle="dashed", label="Difference in Differences"
        )

        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18
        )
        plt.show()

    def plot(self, model="sdid", figsize=(10, 7)):

        result = pd.DataFrame({"actual_y": self.target_y()})
        post_actural_treat = result.loc[self.post_term[0] :, "actual_y"].mean()
        post_point = np.mean(self.Y_post_c.index)

        if model == "sdid":
            result["sdid"] = self.sdid_trajectory()
            time_result = pd.DataFrame(
                {
                    "time": self.Y_pre_c.index,
                    "sdid_weight": np.round(self.hat_lambda, 3),
                }
            )

            pre_point = self.Y_pre_c.index @ self.hat_lambda

            pre_sdid = result["sdid"].head(len(self.hat_lambda)) @ self.hat_lambda
            post_sdid = result.loc[self.post_term[0] :, "sdid"].mean()

            pre_treat = (self.Y_pre_t.T @ self.hat_lambda).values[0]
            counterfuctual_post_treat = pre_treat + (post_sdid - pre_sdid)

            fig, ax = plt.subplots(figsize=figsize)

            result["actual_y"].plot(
                ax=ax, color="blue", linewidth=1, label="treatment group", alpha=0.6
            )
            result["sdid"].plot(
                ax=ax, color="red", linewidth=1, label="syntetic control", alpha=0.6
            )
            ax.plot(
                [pre_point, post_point],
                [pre_sdid, post_sdid],
                label="",
                marker="o",
                color="red",
            )
            ax.plot(
                [pre_point, post_point],
                [pre_treat, post_actural_treat],
                label="",
                marker="o",
                color="blue",
            )
            ax.plot(
                [pre_point, post_point],
                [pre_treat, counterfuctual_post_treat],
                label="",
                marker="o",
                color="blue",
                linewidth=1,
                linestyle="dashed",
                alpha=0.3,
            )

            ax.axvline(
                x=self.pre_term[1],
                linewidth=1,
                linestyle="dashed",
                color="black",
                alpha=0.3,
            )

            ax2 = ax.twinx()
            ax2.bar(
                time_result["time"],
                time_result["sdid_weight"],
                color="#ff7f00",
                label="time weight",
                width=1.0,
                alpha=0.6,
            )
            ax2.set_ylim(0, 3)
            ax2.axis("off")
            ax.set_title(
                f"Synthetic Difference in Differences : tau {round( post_actural_treat - counterfuctual_post_treat,4)}"
            )
            ax.legend()
            plt.show()

        elif model == "sc":
            result["sc"] = self.sc_potentical_outcome()

            pre_sc = result.loc[: self.pre_term[1], "sc"].mean()
            post_sc = result.loc[self.post_term[0] :, "sc"].mean()

            pre_treat = self.Y_pre_t.mean()
            counterfuctual_post_treat = post_sc

            fig, ax = plt.subplots(figsize=figsize)

            result["actual_y"].plot(
                ax=ax, color="blue", linewidth=1, label="treatment group", alpha=0.6
            )
            result["sc"].plot(
                ax=ax, color="red", linewidth=1, label="syntetic control", alpha=0.6
            )

            ax.annotate(
                "",
                xy=(post_point, post_actural_treat),
                xytext=(post_point, counterfuctual_post_treat),
                arrowprops=dict(arrowstyle="-|>", color="black"),
            )

            ax.axvline(
                x=self.pre_term[1],
                linewidth=1,
                linestyle="dashed",
                color="black",
                alpha=0.3,
            )
            ax.set_title(
                f"Synthetic Control Methods : tau {round( post_actural_treat - counterfuctual_post_treat,4)}"
            )
            ax.legend()
            plt.show()

        elif model == "did":
            Y_pre_t = self.Y_pre_t.copy()
            Y_post_t = self.Y_post_t.copy()
            if type(Y_pre_t) != pd.DataFrame:
                Y_pre_t = pd.DataFrame(Y_pre_t)

            if type(Y_post_t) != pd.DataFrame:
                Y_post_t = pd.DataFrame(Y_post_t)

            result["did"] = self.df[self.control].mean(axis=1)
            pre_point = np.mean(self.Y_pre_c.index)

            pre_did = result.loc[: self.pre_term[1], "did"].mean()
            post_did = result.loc[self.post_term[0] :, "did"].mean()

            pre_treat = Y_pre_t.mean(axis=1).mean()
            counterfuctual_post_treat = pre_treat + (post_did - pre_did)

            fig, ax = plt.subplots(figsize=figsize)

            result["actual_y"].plot(
                ax=ax, color="blue", linewidth=1, label="treatment group", alpha=0.6
            )
            result["did"].plot(
                ax=ax, color="red", linewidth=1, label="control", alpha=0.6
            )

            ax.plot(
                [pre_point, post_point],
                [pre_did, post_did],
                label="",
                marker="o",
                color="red",
            )
            ax.plot(
                [pre_point, post_point],
                [pre_treat, post_actural_treat],
                label="",
                marker="o",
                color="blue",
            )
            ax.plot(
                [pre_point, post_point],
                [pre_treat, counterfuctual_post_treat],
                label="",
                marker="o",
                color="blue",
                linewidth=1,
                linestyle="dashed",
                alpha=0.3,
            )

            ax.axvline(
                x=self.pre_term[1],
                linewidth=1,
                linestyle="dashed",
                color="black",
                alpha=0.3,
            )
            ax.set_title(
                f"Difference in Differences : tau {round( post_actural_treat - counterfuctual_post_treat,4)}"
            )
            ax.legend()
            plt.show()

        else:
            print(f"sorry: {model} is not yet available.")

    def hat_tau(self, model="sdid"):
        """
        return ATT
        """
        result = pd.DataFrame({"actual_y": self.target_y()})
        post_actural_treat = result.loc[self.post_term[0] :, "actual_y"].mean()

        if model == "sdid":
            result["sdid"] = self.sdid_trajectory()
            time_result = pd.DataFrame(
                {
                    "time": self.Y_pre_c.index,
                    "sdid_weight": np.round(self.hat_lambda, 3),
                }
            )

            pre_point = self.Y_pre_c.index @ self.hat_lambda
            post_point = np.mean(self.Y_post_c.index)

            pre_sdid = result["sdid"].head(len(self.hat_lambda)) @ self.hat_lambda
            post_sdid = result.loc[self.post_term[0] :, "sdid"].mean()

            pre_treat = (self.Y_pre_t.T @ self.hat_lambda).values[0]
            counterfuctual_post_treat = pre_treat + (post_sdid - pre_sdid)

        elif model == "sc":
            result["sc"] = self.sc_potentical_outcome()
            post_sc = result.loc[self.post_term[0] :, "sc"].mean()
            counterfuctual_post_treat = post_sc

        elif model == "did":
            Y_pre_t = self.Y_pre_t.copy()
            Y_post_t = self.Y_post_t.copy()
            if type(Y_pre_t) != pd.DataFrame:
                Y_pre_t = pd.DataFrame(Y_pre_t)

            if type(Y_post_t) != pd.DataFrame:
                Y_post_t = pd.DataFrame(Y_post_t)

            # actural treat
            post_actural_treat = (
                Y_post_t.mean(axis=1).mean() - Y_pre_t.mean(axis=1).mean()
            )
            counterfuctual_post_treat = (
                self.Y_post_c.mean(axis=1).mean() - self.Y_pre_c.mean(axis=1).mean()
            )

        return post_actural_treat - counterfuctual_post_treat


if __name__ == "__main__":
    from sample_data import fetch_CaliforniaSmoking

    df = fetch_CaliforniaSmoking()

    PRE_TEREM = [1970, 1979]
    POST_TEREM = [1980, 1988]
    TREATMENT = ["California"]

    sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
    sdid.base_zeta
    sdid.fit(zeta_type="base")
    print(sdid.base_zeta)
    print(sdid.zeta)
    sdid.fit(zeta_type="grid_search")
    print(sdid.zeta)
    sdid.fit(zeta_type="bayesian_opt")
    print(sdid.zeta)
