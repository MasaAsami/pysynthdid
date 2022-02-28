import pandas as pd
import numpy as np
from tqdm import tqdm


class Variance(object):
    def estimate_variance(self, algo="placebo", replications=200):
        """
        # algo
        - placebo
        ## The following algorithms are omitted because they are not practical.
        - bootstrap
        - jackknife
        """

        if algo == "placebo":
            Y_pre_c = self.Y_pre_c.copy()
            Y_post_c = self.Y_post_c.copy()
            assert self.n_treat < Y_pre_c.shape[1]
            control_names = Y_pre_c.columns

            result_tau_sdid = []
            result_tau_sc = []
            result_tau_did = []
            for i in tqdm(range(replications)):
                # setup
                np.random.seed(seed=self.random_seed + i)
                placebo_t = np.random.choice(control_names, self.n_treat, replace=False)
                placebo_c = [col for col in control_names if col not in placebo_t]
                pla_Y_pre_t = Y_pre_c[placebo_t]
                pla_Y_post_t = Y_post_c[placebo_t]
                pla_Y_pre_c = Y_pre_c[placebo_c]
                pla_Y_post_c = Y_post_c[placebo_c]

                pla_result = pd.DataFrame(
                    {
                        "pla_actual_y": pd.concat([pla_Y_pre_t, pla_Y_post_t]).mean(
                            axis=1
                        )
                    }
                )
                post_placebo_treat = pla_result.loc[
                    self.post_term[0] :, "pla_actual_y"
                ].mean()

                # estimation
                ## sdid
                pla_zeta = self.est_zeta(pla_Y_pre_c)

                pla_hat_omega = self.est_omega(pla_Y_pre_c, pla_Y_pre_t, pla_zeta)
                pla_hat_lambda = self.est_lambda(pla_Y_pre_c, pla_Y_post_c)
                ## sc
                pla_hat_omega_ADH = self.est_omega_ADH(pla_Y_pre_c, pla_Y_pre_t)

                # prediction
                ## sdid
                pla_hat_omega = pla_hat_omega[:-1]
                pla_Y_c = pd.concat([pla_Y_pre_c, pla_Y_post_c])
                n_features = pla_Y_pre_c.shape[1]
                start_w = np.repeat(1 / n_features, n_features)

                _intercept = (start_w - pla_hat_omega) @ pla_Y_pre_c.T @ pla_hat_lambda

                pla_result["sdid"] = pla_Y_c.dot(pla_hat_omega) + _intercept

                ## sc
                pla_result["sc"] = pla_Y_c.dot(pla_hat_omega_ADH)

                # cal tau
                ## sdid
                pre_sdid = pla_result["sdid"].head(len(pla_hat_lambda)) @ pla_hat_lambda
                post_sdid = pla_result.loc[self.post_term[0] :, "sdid"].mean()

                pre_treat = (pla_Y_pre_t.T @ pla_hat_lambda).values[0]
                sdid_counterfuctual_post_treat = pre_treat + (post_sdid - pre_sdid)

                result_tau_sdid.append(
                    post_placebo_treat - sdid_counterfuctual_post_treat
                )

                ## sc
                sc_counterfuctual_post_treat = pla_result.loc[
                    self.post_term[0] :, "sc"
                ].mean()
                result_tau_sc.append(post_placebo_treat - sc_counterfuctual_post_treat)

                # did
                did_post_actural_treat = (
                    post_placebo_treat
                    - pla_result.loc[: self.pre_term[1], "pla_actual_y"].mean()
                )
                did_counterfuctual_post_treat = (
                    pla_Y_post_c.mean(axis=1).mean() - pla_Y_pre_c.mean(axis=1).mean()
                )
                result_tau_did.append(
                    did_post_actural_treat - did_counterfuctual_post_treat
                )

            return (
                np.var(result_tau_sdid),
                np.var(result_tau_sc),
                np.var(result_tau_did),
            )
