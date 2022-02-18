from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import reduce, partial
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization


def fetch_sampledata() -> pd.DataFrame:
    """
    This data is from https://web.stanford.edu/~jhain/synthpage.html
    [Retun]
    pd.DataFrame
    """
    _raw = pd.read_csv("../sample_data/MLAB_data.txt", sep="\t", header=None)

    _raw.columns = [
        "Alabama",
        "Arkansas",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Georgia",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Mexico",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "California",
    ]

    _raw.index = [i for i in range(1962, 2001)]

    return _raw.loc[1970:]


def cal_did_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, pre_cal=False):
    """
    pre_cal: 介入前も計算するか否か
    """
    if type(Y_pre_t) != pd.DataFrame:
        Y_pre_t = pd.DataFrame(Y_pre_t)
    
    if not pre_cal:
        return (
            Y_pre_t.mean(axis=1).mean()
            + Y_post_c.mean(axis=1).mean()
            - Y_pre_c.mean(axis=1).mean()
        )
    else:
        return Y_pre_t.mean(axis=1).mean() ,(
            Y_pre_t.mean(axis=1).mean()
            + Y_post_c.mean(axis=1).mean()
            - Y_pre_c.mean(axis=1).mean()
        )

def cal_ADHsc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, pre_cal=False, plot_params=False):
    hat_omega_ADH = est_omega_ADH(Y_pre_c, Y_pre_t)
    if plot_params:
        print(pd.DataFrame({"states":Y_pre_c.columns, "sdid_weight":np.round(hat_omega_ADH,3) }) )
    if not pre_cal:
        return Y_post_c.dot(hat_omega_ADH) 
    else:
        return pd.concat([Y_pre_c, Y_post_c]).dot(hat_omega_ADH) 


def cal_sc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None, pre_cal=False):
    Y_post_c_intercept = Y_post_c.copy()
    Y_post_c_intercept["intercept"] = 1

    if zeta == None:
        n_treat = Y_pre_t.shape[1]  
        n_post_term = len(Y_post_c)
        zeta = est_zeta(Y_pre_c, n_treat, n_post_term)

    hat_omega = est_omega(Y_pre_c, Y_pre_t, zeta=zeta)
    if not pre_cal:
        return Y_post_c_intercept @ hat_omega
    else:
        Y_pre_c_intercept = Y_pre_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        return pd.concat([Y_pre_c_intercept, Y_post_c_intercept]) @ hat_omega


def cal_SDID_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None, pre_cal=False, plot_params=False):

    Y_pre_c_intercept = Y_pre_c.copy()
    Y_post_c_intercept = Y_post_c.copy()
    Y_pre_c_intercept["intercept"] = 1
    Y_post_c_intercept["intercept"] = 1

    if zeta == None:
        n_treat = Y_pre_t.shape[1]  
        n_post_term = len(Y_post_c)
        zeta = est_zeta(Y_pre_c, n_treat, n_post_term)

    hat_omega = est_omega(Y_pre_c, Y_pre_t, zeta=zeta)
    hat_lambda = est_lambda(Y_pre_c, Y_post_c)
    if plot_params:
        print(pd.DataFrame({"states":Y_pre_c_intercept.columns, "sdid_weight":np.round(hat_omega,3) }) )
        print(pd.DataFrame({"time":Y_pre_c_intercept.index, "sdid_weight":np.round(hat_lambda,3) }) )
    
    base_sc = Y_post_c_intercept @ hat_omega
    lambda_effect  = (Y_pre_t.T @ hat_lambda).values[0]
    sc_pretrend_with_timeweighted = Y_pre_c_intercept @ hat_omega @ hat_lambda

    post_outcome = base_sc + lambda_effect - sc_pretrend_with_timeweighted
    
    if not pre_cal:
        return post_outcome
    else:
        return pd.concat([Y_pre_c_intercept.dot(hat_omega), post_outcome],axis=0) 



def est_zeta(Y_pre_c, n_treat, n_post_term):
    return (n_treat * n_post_term) ** (1 / 4) * Y_pre_c.std().std()


def l2_loss(W, X, y, zeta, nrow) -> float:
    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis=1)
    _X = X.copy()
    _X["intersept"] = 1
    return np.sum((y - _X.dot(W)) ** 2) + nrow * zeta ** 2 * np.sum(W ** 2)


def mse_loss(W, X, y, intersept=True) -> float:
    if type(y) == pd.core.frame.DataFrame:
        y = y.mean(axis=1)
    _X = X.copy()
    if intersept:
        _X["intersept"] = 1
    return np.mean(np.sqrt((y - _X.dot(W)) ** 2))


def est_omega(Y_pre_c, Y_pre_t, zeta=1):
    n_features = Y_pre_c.shape[1]
    nrow = Y_pre_c.shape[0]

    _w = np.repeat(1 / n_features, n_features)
    _w0 = 1

    start_w = np.append(_w, _w0)

    if type(Y_pre_t) == pd.core.frame.DataFrame:
        Y_pre_t = Y_pre_t.mean(axis=1)
        Y_pre_t.columns = "treatment_group"

    # Required to have non negative values
    max_bnd = abs(Y_pre_t.mean()) * 2
    w_bnds = tuple(
        (0, 1) if i < n_features else (max_bnd * -1, max_bnd)
        for i in range(n_features + 1)
    )

    caled_w = fmin_slsqp(
        partial(l2_loss, X=Y_pre_c, y=Y_pre_t, zeta=zeta, nrow=nrow),
        start_w,
        f_eqcons=lambda x: np.sum(x[:n_features]) - 1,
        bounds=w_bnds,
        disp=False,
    )

    return caled_w

def est_omega_ADH(Y_pre_c, Y_pre_t):
    n_features = Y_pre_c.shape[1]
    nrow = Y_pre_c.shape[0]

    _w = np.repeat(1 / n_features, n_features)

    if type(Y_pre_t) == pd.core.frame.DataFrame:
        Y_pre_t = Y_pre_t.mean(axis=1)
        Y_pre_t.columns = "treatment_group"

    # normalized
    # temp_df = pd.concat([Y_pre_c, Y_pre_t], axis=1)
    # ss = StandardScaler()
    # ss_df = pd.DataFrame(ss.fit_transform(temp_df) , columns=temp_df.columns, index=temp_df.index)

    # ss_Y_pre_c = ss_df.iloc[:,:-1]
    # ss_Y_pre_t = ss_df.iloc[:,-1]

    # Required to have non negative values
    w_bnds = tuple((0, 1) for i in range(n_features ))

    caled_w = fmin_slsqp(
        partial(mse_loss, X=Y_pre_c, y=Y_pre_t, intersept=False),
        _w,
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=w_bnds,
        disp=False,
    )

    return caled_w


def _zeta_given_cv_loss(Y_pre_c, Y_pre_t, zeta, cv=5):
    nrow = Y_pre_c.shape[0]
    kf = KFold(n_splits=cv)
    loss_result = []
    nf_result = []
    for train_index, test_index in kf.split(Y_pre_c, Y_pre_t):
        train_w = est_omega(Y_pre_c.iloc[train_index], Y_pre_t.iloc[train_index], zeta)

        nf_result.append(np.sum(np.round(np.abs(train_w), 3) > 0) - 1)

        loss_result.append(
            mse_loss(train_w, Y_pre_c.iloc[test_index], Y_pre_t.iloc[test_index])
        )
    return np.mean(loss_result), np.mean(nf_result)


def gread_search_zeta(Y_pre_c, Y_pre_t, n_treat, n_post_term, cv=5, candidate_zata=[]):

    if len(candidate_zata) == 0:
        base_zeta = est_zeta(Y_pre_c, n_treat, n_post_term)

        for _z in np.linspace(0.1, base_zeta * 2, 30):
            candidate_zata.append(_z)

        candidate_zata.append(base_zeta)
        candidate_zata.append(0)

        candidate_zata = sorted(candidate_zata)

        result_loss_dict = {}
        result_nf_dict = {}

    print("cv: zeta")
    for _zeta in tqdm(candidate_zata):
        result_loss_dict[_zeta], result_nf_dict[_zeta] = _zeta_given_cv_loss(
            Y_pre_c, Y_pre_t, _zeta
        )

    loss_sorted = sorted(result_loss_dict.items(), key=lambda x: x[1])

    return loss_sorted[0]


def bayes_opt_zeta(
    Y_pre_c,
    Y_pre_t,
    n_treat,
    n_post_term,
    cv=5,
    init_points=10,
    n_iter=10,
    zeta_max=None,
    zeta_min=None,
):
    base_zeta = est_zeta(Y_pre_c, n_treat, n_post_term)

    if zeta_max == None:
        zeta_max = base_zeta * 2

    if zeta_min == None:
        zeta_min = 0.01

    pbounds = {"zeta": (zeta_min, zeta_max)}

    optimizer = BayesianOptimization(
        f=partial(_zeta_given_cv_loss_inverse, Y_pre_c=Y_pre_c, Y_pre_t=Y_pre_t),
        pbounds=pbounds,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    optimizer.max["params"]["zeta"]

    return (optimizer.max["params"]["zeta"], optimizer.max["target"] * -1)


def est_lambda(Y_pre_c, Y_post_c):
    Y_pre_c_T = Y_pre_c.T
    Y_post_c_T = Y_post_c.T

    n_pre_term = Y_pre_c_T.shape[1]

    _lambda = np.repeat(1 / n_pre_term, n_pre_term)
    _lambda0 = 1

    start_lambda = np.append(_lambda, _lambda0)

    if type(Y_post_c_T) == pd.core.frame.DataFrame:
        Y_post_c_T = Y_post_c_T.mean(axis=1)
        Y_post_c_T.columns = "mean_Y_post"

    # Required to have non negative values
    max_bnd = abs(Y_post_c_T.mean()) * 2
    lambda_bnds = tuple(
        (0, 1) if i < n_pre_term else (max_bnd * -1, max_bnd)
        for i in range(n_pre_term + 1)
    )

    caled_lambda = fmin_slsqp(
        partial(l2_loss, X=Y_pre_c_T, y=Y_post_c_T, zeta=0, nrow=0),
        start_lambda,
        f_eqcons=lambda x: np.sum(x[:n_pre_term]) - 1,
        bounds=lambda_bnds,
        disp=False,
    )

    return caled_lambda[:n_pre_term]


def _zeta_given_cv_loss_inverse(Y_pre_c, Y_pre_t, zeta, cv=5):
    return -1 * _zeta_given_cv_loss(Y_pre_c, Y_pre_t, zeta, cv)[0]


def sample_data_model_comparison(df, treatment):
    contorol = [col for col in df.columns if col not in [treatment]]
    #PRE_TEREM = [1970, 1987]
    #POST_TEREM = [1988, 2000]
    PRE_TEREM = [1970, 1979]
    POST_TEREM = [1980, 1988]
    Y_pre_c = df.loc[PRE_TEREM[0] : PRE_TEREM[1], contorol]
    Y_pre_t = df.loc[PRE_TEREM[0] : PRE_TEREM[1], [treatment]]

    Y_post_c = df.loc[POST_TEREM [0] : POST_TEREM [1], contorol]
    Y_post_t = df.loc[POST_TEREM [0] : POST_TEREM [1], [treatment]]

    result_df = Y_post_t.copy()

    result_df["did"] = cal_did_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c)
    result_df["adh_sc"] = cal_ADHsc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c)
    #result_df["sc"] = cal_sc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None)
    result_df["sdid"] = cal_SDID_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None)

    mean_y = result_df.mean()
    mean_y = pd.DataFrame(mean_y).T.rename(columns={treatment:"true_Y"})
    mean_y.index = [treatment]
    return mean_y

def sample_data_model_plot(df, treatment = 'California'):
    contorol = [col for col in df.columns if col not in [treatment]]
    #PRE_TEREM = [1970, 1988]
    #POST_TEREM = [1989, 2000]
    PRE_TEREM = [1970, 1979]
    POST_TEREM = [1980, 1988]
    Y_pre_c = df.loc[PRE_TEREM[0] : PRE_TEREM[1], contorol]
    Y_pre_t = df.loc[PRE_TEREM[0] : PRE_TEREM[1], [treatment]]

    Y_post_c = df.loc[POST_TEREM [0] : POST_TEREM [1], contorol]
    Y_post_t = df.loc[POST_TEREM [0] : POST_TEREM [1], [treatment]]

    result_df = pd.concat([Y_pre_t, Y_post_t], axis=0)

    result_df.loc[PRE_TEREM, "did"] = cal_did_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, pre_cal=True)[0]
    result_df.loc[POST_TEREM, "did"] = cal_did_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, pre_cal=True)[1]
    result_df = result_df.interpolate(limit_direction='both')
    result_df["adh_sc"] = cal_ADHsc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, pre_cal=True)
    #result_df["sc"] = cal_sc_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None, pre_cal=True)
    result_df["sdid"] = cal_SDID_potentical_outcome(Y_pre_c, Y_pre_t, Y_post_c, zeta=None, pre_cal=True)

    hat_omega_ADH = est_omega_ADH(Y_pre_c, Y_pre_t)


    result_df.plot()
    plt.show()


    mean_y = result_df.mean()
    mean_y = pd.DataFrame(mean_y).T.rename(columns={treatment:"true_Y"})
    mean_y.index = [treatment]
    return mean_y


if __name__ == '__main__' :
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = fetch_sampledata()
    state_list = df.columns
    result_df_list = []
    for _state in tqdm(state_list):
        result_df_list.append(sample_data_model_comparison(df, _state))
    
    result_df = pd.concat(result_df_list)

    result_rmse = pd.DataFrame()

    result_rmse["did"] = result_df.apply(lambda x:  np.sqrt((x["true_Y"] - x["did"])**2), axis=1)
    result_rmse["adh_sc"] = result_df.apply(lambda x:  np.sqrt((x["true_Y"] - x["adh_sc"])**2), axis=1)
    result_rmse["sc"] = result_df.apply(lambda x:  np.sqrt((x["true_Y"] - x["sc"])**2), axis=1)
    result_rmse["sdid"] = result_df.apply(lambda x:  np.sqrt((x["true_Y"] - x["sdid"])**2), axis=1)

    result_rmse["is_California"] = np.where(result_rmse.index=="California", True, False)

    fig = plt.figure()
    plt.style.use('ggplot')
    ax = fig.add_subplot(1, 1, 1) 
    _x = np.linspace(0, 50, 30)
    _y = _x
    sns.scatterplot(data=result_rmse, x="sdid", y="adh_sc", hue="is_California", style="is_California", ax = ax)
    ax.plot(_x, _y, color='black',  linestyle='solid',linewidth = 0.5)
    ax.set_xlabel("Synthetic Difference in Difference")
    ax.set_ylabel("Synthetic Contorol")
    #ax.set_xlim(0, 50)
    #ax.set_ylim(0, 50)
    plt.show()



