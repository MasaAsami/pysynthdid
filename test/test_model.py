from operator import index
from pyexpat import model
import sys
import os
from scipy.stats import spearmanr

sys.path.append(os.path.abspath("../"))

import pandas as pd
import numpy as np
import pytest

from synthdid.model import SynthDID
from synthdid.sample_data import fetch_CaliforniaSmoking


class TestModelSynth(object):
    def test_params_with_originalpaper(self):
        """
        Original Paper (see: Arkhangelsky, Dmitry, et al. Synthetic difference in differences. No. w25532. National Bureau of Economic Research, 2019. https://arxiv.org/abs/1812.09970)
        """
        test_df = fetch_CaliforniaSmoking()
        test_omega = pd.read_csv("test_data/omega_CalifolinaSmoking.csv")
        test_lambda = pd.read_csv("test_data/lambda_CalifolinaSmoking.csv")
        PRE_TEREM = [1970, 1988]
        POST_TEREM = [1989, 2000]

        TREATMENT = ["California"]

        sdid = SynthDID(test_df, PRE_TEREM, POST_TEREM, TREATMENT)

        sdid.fit(zeta_type="base")

        hat_omega_sdid, hat_lambda_sdid = sdid.estimated_params()
        hat_omega = sdid.estimated_params(model="sc")

        omega_result = pd.merge(
            test_omega, hat_omega_sdid, left_on="state", right_on="features", how="left"
        )
        omega_result = pd.merge(
            omega_result, hat_omega, left_on="state", right_on="features", how="left"
        )
        omega_result["random"] = 1 / len(omega_result)

        error_random_omega = np.sqrt(
            omega_result.eval("omega_sdid - random") ** 2
        ).sum()

        # Classic SC Result
        error_sc_omega = np.sqrt(omega_result.eval("omega_ADH - sc_weight") ** 2).sum()
        assert error_sc_omega < error_random_omega

        adh_corr, _p = spearmanr(omega_result["omega_ADH"], omega_result["sc_weight"])
        assert adh_corr >= 0.9

        # SDID Result
        error_sdid_omega = np.sqrt(
            omega_result.eval("omega_sdid - sdid_weight") ** 2
        ).sum()
        assert error_sdid_omega < error_random_omega

        sdid_corr, _p = spearmanr(
            omega_result["omega_sdid"], omega_result["sdid_weight"]
        )
        assert sdid_corr >= 0.9

        lambda_result = pd.merge(
            test_lambda, hat_lambda_sdid, left_on="year", right_on="time", how="left"
        )
        lambda_result["random"] = 1 / len(lambda_result)

        # lambda test
        error_random_lambda = np.sqrt(
            lambda_result.eval("lambda_sdid - random") ** 2
        ).sum()
        error_sdid_lambda = np.sqrt(
            lambda_result.eval("lambda_sdid - sdid_weight") ** 2
        ).sum()
        assert error_sdid_lambda < error_random_lambda

        sdid_corr, _p = spearmanr(
            lambda_result["lambda_sdid"], lambda_result["sdid_weight"]
        )
        assert sdid_corr >= 0.9

    def test_multi_treatment(self):
        """
        トリートメントの数が変わればzetaも変わる
        """

        test_df = fetch_CaliforniaSmoking()
        PRE_TEREM = [1970, 1979]
        POST_TEREM = [1980, 1988]

        treatment = [col for i, col in enumerate(test_df.columns) if i % 2 == 0]

        multi_sdid = SynthDID(test_df, PRE_TEREM, POST_TEREM, treatment)
        multi_sdid.fit(zeta_type="base")

        hat_omega_sdid, hat_lambda_sdid = multi_sdid.estimated_params()
        hat_omega = multi_sdid.estimated_params(model="sc")

        assert (
            np.round(
                hat_omega_sdid.query("features != 'intercept'")["sdid_weight"].sum(), 2
            )
            == 1.0
        )

        assert np.round(hat_omega["sc_weight"].sum(), 2) == 1.0

        treatment2 = [col for i, col in enumerate(test_df.columns) if i % 3 == 0]

        multi_sdid2 = SynthDID(test_df, PRE_TEREM, POST_TEREM, treatment2)
        multi_sdid2.fit(zeta_type="base")

        assert multi_sdid2.zeta != multi_sdid.zeta

    def test_short_preterm(self):
        """
        極端なケースを目してみる
        """

        test_df = fetch_CaliforniaSmoking()
        pre_term = [1970, 1971]
        post_term = [1972, 2000]

        treatment = ["California"]

        multi_sdid = SynthDID(test_df, pre_term, post_term, treatment)
        multi_sdid.fit(zeta_type="base")

        hat_omega_sdid, hat_lambda_sdid = multi_sdid.estimated_params()
        hat_omega = multi_sdid.estimated_params(model="sc")

        assert (
            np.round(
                hat_omega_sdid.query("features != 'intercept'")["sdid_weight"].sum(), 2
            )
            == 1.0
        )

        assert np.round(hat_omega["sc_weight"].sum(), 2) == 1.0
