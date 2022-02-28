import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Plot(object):
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
                x=(self.pre_term[1] + self.post_term[0]) * 0.5,
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
                x=(self.pre_term[1] + self.post_term[0]) * 0.5,
                linewidth=1,
                linestyle="dashed",
                color="black",
                alpha=0.3,
            )
            ax.set_title(
                f"Synthetic Control Method : tau {round( post_actural_treat - counterfuctual_post_treat,4)}"
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
                x=(self.pre_term[1] + self.post_term[0]) * 0.5,
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
