import numpy as np
import pandas as pd
import talib as ta
import itertools as ittl
from husfort.qcalendar import CCalendar
from typedef import (
    CCfgFactorMTM, CCfgFactorSKEW, CCfgFactorKURT, CCfgFactorRS,
    CCfgFactorBASIS, CCfgFactorTS,
    CCfgFactorS0BETA, CCfgFactorS1BETA,
    CCfgFactorCBETA, CCfgFactorIBETA, CCfgFactorPBETA,
    CCfgFactorCTP, CCfgFactorCVP, CCfgFactorCSP,
    CCfgFactorCTR, CCfgFactorCVR, CCfgFactorCSR,
    CCfgFactorCOV,
    CCfgFactorNOI, CCfgFactorNDOI, CCfgFactorWNOI, CCfgFactorWNDOI, CCfgFactorSPDWEB,
    CCfgFactorAMP, CCfgFactorEXR, CCfgFactorSMT, CCfgFactorRWTC, CCfgFactorTAILS, CCfgFactorHEADS,
    CCfgFactorTOPS,
    CCfgFactorTA, CCfgFactorSIZE, CCfgFactorHR, CCfgFactorSR, CCfgFactorLIQUIDITY, CCfgFactorVSTD
)
from solutions.factor import CFactorRaw

"""
-----------------------
Part I: Some math tools
-----------------------
"""


def cal_rolling_corr(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    df["xy"] = (df[x] * df[y]).rolling(window=rolling_window).mean()
    df["xx"] = (df[x] * df[x]).rolling(window=rolling_window).mean()
    df["yy"] = (df[y] * df[y]).rolling(window=rolling_window).mean()
    df["x"] = df[x].rolling(window=rolling_window).mean()
    df["y"] = df[y].rolling(window=rolling_window).mean()

    df["cov_xy"] = df["xy"] - df["x"] * df["y"]
    df["cov_xx"] = df["xx"] - df["x"] * df["x"]
    df["cov_yy"] = df["yy"] - df["y"] * df["y"]

    # due to float number precision, cov_xx or cov_yy could be slightly negative
    df.loc[np.abs(df["cov_xx"]) <= 1e-10, "cov_xx"] = 0
    df.loc[np.abs(df["cov_yy"]) <= 1e-10, "cov_yy"] = 0

    df["sqrt_cov_xx_yy"] = np.sqrt(df["cov_xx"] * df["cov_yy"])
    s = df[["cov_xy", "sqrt_cov_xx_yy"]].apply(
        lambda z: 0 if z["sqrt_cov_xx_yy"] == 0 else z["cov_xy"] / z["sqrt_cov_xx_yy"], axis=1
    )
    return s


def cal_rolling_beta(df: pd.DataFrame, x: str, y: str, rolling_window: int) -> pd.Series:
    df["xy"] = (df[x] * df[y]).rolling(window=rolling_window).mean()
    df["xx"] = (df[x] * df[x]).rolling(window=rolling_window).mean()
    df["x"] = df[x].rolling(window=rolling_window).mean()
    df["y"] = df[y].rolling(window=rolling_window).mean()
    df["cov_xy"] = df["xy"] - df["x"] * df["y"]
    df["cov_xx"] = df["xx"] - df["x"] * df["x"]
    s = df["cov_xy"] / df["cov_xx"]
    return s


def cal_top_corr(sub_data: pd.DataFrame, x: str, y: str, sort_var: str, top_size: int, ascending: bool = False):
    sorted_data = sub_data.sort_values(by=sort_var, ascending=ascending)
    top_data = sorted_data.head(top_size)
    r = top_data[[x, y]].corr(method="spearman").at[x, y]
    return r


def auto_weight_sum(x: pd.Series) -> float:
    weight = x.abs() / x.abs().sum()
    return x @ weight


"""
---------------------------------------------------
Part II: factor class from different configuration
---------------------------------------------------
"""


class CFactorMTM(CFactorRaw):
    def __init__(self, cfg: CCfgFactorMTM, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = major_data["return_c_major"].rolling(window=win).sum()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorSKEW(CFactorRaw):
    def __init__(self, cfg: CCfgFactorSKEW, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = -major_data["return_c_major"].rolling(window=win).skew()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorKURT(CFactorRaw):
    def __init__(self, cfg: CCfgFactorKURT, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = major_data["return_c_major"].rolling(window=win).kurt()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorRS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorRS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        __min_win = 5
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "stock"],
        )
        adj_data["stock"] = adj_data["stock"].ffill(limit=__min_win).fillna(0)
        for win in self.cfg.wins:
            rspa = f"{self.factor_class}PA{win:03d}"
            rsla = f"{self.factor_class}LA{win:03d}"

            ma = adj_data["stock"].rolling(window=win).mean()
            s = adj_data["stock"] / ma
            s[s == np.inf] = np.nan  # some maybe resulted from divided by Zero
            adj_data[rspa] = 1 - s

            la = adj_data["stock"].shift(win)
            s = adj_data["stock"] / la
            s[s == np.inf] = np.nan  # some maybe resulted from divided by Zero
            adj_data[rsla] = 1 - s
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorBASIS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorBASIS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "basis_rate"],
        )
        for win in self.cfg.wins:
            f0 = f"{self.factor_class}{win:03d}"
            f1 = f"{self.factor_class}D{win:03d}"
            adj_data[f0] = adj_data["basis_rate"].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            adj_data[f1] = adj_data["basis_rate"] - adj_data[f0]
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorTS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorTS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_roll_return(x: pd.Series, ticker_n: str, ticker_d: str, prc_n: str, prc_d: str):
        if x[ticker_n] == "" or x[ticker_d] == "":
            return np.nan
        if x[prc_d] > 0:
            cntrct_d, cntrct_n = x[ticker_d].split(".")[0], x[ticker_n].split(".")[0]
            month_d, month_n = int(cntrct_d[-2:]), int(cntrct_n[-2:])
            dlt_month = month_d - month_n
            dlt_month = dlt_month + (12 if dlt_month <= 0 else 0)
            return (x[prc_n] / x[prc_d] - 1) / dlt_month * 12 * 100
        else:
            return np.nan

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "ticker_minor", "close_major", "close_minor"],
        )
        adj_data[["ticker_major", "ticker_minor"]] = adj_data[["ticker_major", "ticker_minor"]].fillna("")
        adj_data["ts"] = adj_data.apply(
            self.cal_roll_return,
            args=("ticker_major", "ticker_minor", "close_major", "close_minor"),
            axis=1,
        )
        for win in self.cfg.wins:
            f0 = f"{self.factor_class}{win:03d}"
            f1 = f"{self.factor_class}D{win:03d}"
            adj_data[f0] = adj_data["ts"].rolling(window=win, min_periods=int(2 * win / 3)).mean()
            adj_data[f1] = adj_data["ts"] - adj_data[f0]
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class __CFactorBETA(CFactorRaw):
    @staticmethod
    def merge_xy(x_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
        adj_data = pd.merge(left=x_data, right=y_data, how="left", on="trade_date")
        return adj_data

    def betas_from_wins(self, wins: list[int], input_data: pd.DataFrame, x: str, y: str):
        __prefix0 = f"{self.factor_class}{wins[0]:03d}"
        f0 = f"{__prefix0}"
        for i, win in enumerate(wins):
            fi = f"{self.factor_class}{win:03d}"
            input_data[fi] = cal_rolling_beta(df=input_data, x=x, y=y, rolling_window=win)
            if i > 0:
                fid = f"{__prefix0}D{win:03d}"
                input_data[fid] = input_data[f0] - input_data[fi]
        return 0

    def res_from_wins(self, wins: list[int], input_data: pd.DataFrame, x: str, y: str):
        for i, win in enumerate(wins):
            b, fi = f"{self.factor_class}{win:03d}", f"{self.factor_class}{win:03d}RES"
            input_data[fi] = input_data[y] - input_data[x] * input_data[b]
        return 0

    def res_std_from_wins(self, wins: list[int], input_data: pd.DataFrame):
        for i, win in enumerate(wins):
            res, fi = f"{self.factor_class}{win:03d}RES", f"{self.factor_class}{win:03d}RESSTD"
            input_data[fi] = input_data[res].rolling(window=win, min_periods=int(win * 0.6)).std()
        return 0


class CFactorS0BETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorS0BETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret = "INH0100_NHF" if self.universe[instru].sectorL0 == "C" else "I881001_WI"
        __y_ret = "return_c_major"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        adj_market_data = self.load_mkt(bgn_date=win_start_date, stp_date=stp_date)
        adj_data = self.merge_xy(
            x_data=adj_major_data[["trade_date", "ticker_major", __y_ret]],
            y_data=adj_market_data[["trade_date", __x_ret]]
        )
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorS1BETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorS1BETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = self.universe[instru].sectorL1, "return_c_major"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        adj_market_data = self.load_mkt(bgn_date=win_start_date, stp_date=stp_date)
        adj_data = self.merge_xy(
            x_data=adj_major_data[["trade_date", "ticker_major", __y_ret]],
            y_data=adj_market_data[["trade_date", __x_ret]]
        )
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorCBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorCBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "pct_chg", "return_c_major"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        adj_forex_data = self.load_forex(bgn_date=win_start_date, stp_date=stp_date)
        adj_data = self.merge_xy(
            x_data=adj_major_data[["trade_date", "ticker_major", __y_ret]],
            y_data=adj_forex_data[["trade_date", __x_ret]]
        )
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorIBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorIBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "cpi_rate", "return_c_major"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        adj_macro_data = self.load_macro(bgn_date=win_start_date, stp_date=stp_date)
        adj_macro_data[__x_ret] = adj_macro_data[__x_ret] / 100
        adj_data = self.merge_xy(
            x_data=adj_major_data[["trade_date", "ticker_major", __y_ret]],
            y_data=adj_macro_data[["trade_date", __x_ret]]
        )
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorPBETA(__CFactorBETA):
    def __init__(self, cfg: CCfgFactorPBETA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __x_ret, __y_ret = "ppi_rate", "return_c_major"
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major"],
        )
        adj_macro_data = self.load_macro(bgn_date=win_start_date, stp_date=stp_date)
        adj_macro_data[__x_ret] = adj_macro_data[__x_ret] / 100
        adj_data = self.merge_xy(
            x_data=adj_major_data[["trade_date", "ticker_major", __y_ret]],
            y_data=adj_macro_data[["trade_date", __x_ret]]
        )
        self.betas_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_from_wins(self.cfg.wins, adj_data, __x_ret, __y_ret)
        self.res_std_from_wins(self.cfg.wins, adj_data)
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class __CFactorCXY(CFactorRaw):
    def cal_rolling_top_corr(
            self,
            raw_data: pd.DataFrame,
            bgn_date: str, stp_date: str,
            x: str, y: str,
            wins: list[int], tops: list[float],
            sort_var: str,
            direction: int = -1,
    ):
        for win, top in ittl.product(wins, tops):
            factor_name = f"{self.factor_class}{win:03d}T{int(top * 10):02d}"
            top_size = int(win * top) + 1
            r_data = {}
            for i, trade_date in enumerate(raw_data.index):
                if trade_date < bgn_date:
                    continue
                elif trade_date >= stp_date:
                    break
                sub_data = raw_data.iloc[i - win + 1: i + 1]
                r_data[trade_date] = cal_top_corr(sub_data, x=x, y=y, sort_var=sort_var, top_size=top_size)
            raw_data[factor_name] = pd.Series(r_data) * direction
        return 0


class CFactorCTP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCTP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins + [2]), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "oi_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        adj_data["aver_oi"] = adj_data["oi_major"].rolling(window=2).mean()
        adj_data["turnover"] = adj_data["vol_major"] / adj_data["aver_oi"]
        x, y = "turnover", "closeI"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCTR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCTR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins + [2]), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "oi_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        adj_data["aver_oi"] = adj_data["oi_major"].rolling(window=2).mean()
        adj_data["turnover"] = adj_data["vol_major"] / adj_data["aver_oi"]
        x, y = "turnover", "return_c_major"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCVP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCVP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        x, y = "vol_major", "closeI"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCVR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCVR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        x, y = "vol_major", "return_c_major"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCSP(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCSP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __near_short_term = 10
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins + [__near_short_term]), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        adj_data["sigma"] = adj_data["return_c_major"].fillna(0).rolling(window=__near_short_term).std()
        x, y = "sigma", "closeI"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCSR(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCSR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __near_short_term = 10
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins + [__near_short_term]), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major", "closeI"],
        )
        adj_data = adj_data.set_index("trade_date")
        adj_data["sigma"] = adj_data["return_c_major"].fillna(0).rolling(window=__near_short_term).std()
        x, y = "sigma", "return_c_major"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class CFactorCOV(__CFactorCXY):
    def __init__(self, cfg: CCfgFactorCOV, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        __near_short_term = 10
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins + [__near_short_term]), -5)
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major", "oi_major"],
        )
        adj_data = adj_data.set_index("trade_date")
        x, y = "vol_major", "oi_major"
        self.cal_rolling_top_corr(
            adj_data,
            bgn_date=bgn_date, stp_date=stp_date,
            x=x, y=y,
            wins=self.cfg.wins, tops=self.cfg.tops,
            sort_var="vol_major",
        )
        adj_data = adj_data.reset_index()
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date=bgn_date)
        return factor_data


class __CFactorMbrPos(CFactorRaw):
    def __init__(self, cfg: CCfgFactorNOI | CCfgFactorNDOI | CCfgFactorWNOI | CCfgFactorWNDOI, **kwargs):
        if cfg.factor_class not in ["NOI", "NDOI", "WNOI", "WNDOI"]:
            raise ValueError(f"factor class - {cfg.factor_class} is illegal")
        self.cfg = cfg
        self.call_weight_sum = cfg.factor_class in ["WNOI", "WNDOI"]
        self.using_diff = cfg.factor_class in ["NDOI", "WNDOI"]
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def __robust_rate(z: pd.Series) -> float:
        return z.iloc[0] / z.iloc[1] * 100 if z.iloc[1] > 0 else np.nan

    def cal_core(self, pos_data: pd.DataFrame, top: int, instru_oi_data: pd.DataFrame) -> pd.DataFrame:
        cntrct_pos_data = pos_data.query("code_type == 0")

        lng_pos_data = cntrct_pos_data[["trade_date", "ts_code", "broker", "long_hld", "long_chg"]].dropna(
            subset=["long_hld", "long_chg"], how="any")
        srt_pos_data = cntrct_pos_data[["trade_date", "ts_code", "broker", "short_hld", "short_chg"]].dropna(
            subset=["short_hld", "short_chg"], how="any")

        lng_rnk_data = lng_pos_data[["trade_date", "ts_code", "long_hld"]].groupby(
            by=["trade_date", "ts_code"]).rank(ascending=False)
        srt_rnk_data = srt_pos_data[["trade_date", "ts_code", "short_hld"]].groupby(
            by=["trade_date", "ts_code"]).rank(ascending=False)

        lng_data = pd.merge(
            left=lng_pos_data, right=lng_rnk_data,
            left_index=True, right_index=True,
            how="left", suffixes=("", "_rnk")
        ).sort_values(by=["trade_date", "ts_code", "long_hld_rnk"], ascending=True)
        lng_data_slc = lng_data.query(f"long_hld_rnk <= {top}")

        srt_data = pd.merge(
            left=srt_pos_data, right=srt_rnk_data,
            left_index=True, right_index=True,
            how="left", suffixes=("", "_rnk")
        ).sort_values(by=["trade_date", "ts_code", "short_hld_rnk"], ascending=True)
        srt_data_slc = srt_data.query(f"short_hld_rnk <= {top}")

        lng_oi_df = pd.pivot_table(
            data=lng_data_slc,
            index="trade_date",
            values="long_chg" if self.using_diff else "long_hld",
            aggfunc=auto_weight_sum if self.call_weight_sum else "sum",
        )
        srt_oi_df = pd.pivot_table(
            data=srt_data_slc,
            index="trade_date",
            values="short_chg" if self.using_diff else "short_hld",
            aggfunc=auto_weight_sum if self.call_weight_sum else "sum",
        )

        noi_df = instru_oi_data.set_index("trade_date").merge(
            right=lng_oi_df, left_index=True, right_index=True, how="left",
        ).merge(
            right=srt_oi_df, left_index=True, right_index=True, how="left",
        )
        if self.using_diff:
            noi_df["noi_sum"] = noi_df["long_chg"]  # - noi_df["short_chg"]
        else:
            noi_df["noi_sum"] = noi_df["long_hld"]  # - noi_df["short_hld"]

        noi_df["net"] = noi_df[["noi_sum", "oi_instru"]].apply(self.__robust_rate, axis=1)
        return noi_df[["net"]]

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)

        # load adj major data as header
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "oi_instru"],
        )

        # load member
        pos_data = self.load_pos(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=[
                "trade_date", "ts_code", "broker",
                "long_hld", "long_chg", "short_hld", "short_chg",
                "code_type"
            ]
        )

        # cal
        res = {}
        for top in self.cfg.tops:
            net_data = self.cal_core(pos_data=pos_data, top=top, instru_oi_data=adj_data[["trade_date", "oi_instru"]])
            for win in self.cfg.wins:
                mp = int(2 * win / 3)
                factor_name = f"{self.factor_class}{win:03d}T{top:02d}"
                res[factor_name] = net_data["net"].rolling(window=win, min_periods=mp).mean()
        res_df = pd.DataFrame(res).reset_index()

        # merge to header
        adj_data = pd.merge(left=adj_data, right=res_df, on="trade_date", how="left")
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorNOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorNOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorNDOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorNDOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorWNOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorWNOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorWNDOI(__CFactorMbrPos):
    def __init__(self, cfg: CCfgFactorWNDOI, **kwargs):
        super().__init__(cfg=cfg, **kwargs)


class CFactorSPDWEB(CFactorRaw):
    def __init__(self, cfg: CCfgFactorSPDWEB, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_spdweb(self, trade_date_data: pd.DataFrame) -> pd.Series:
        n = len(trade_date_data)
        res = {}
        for prop, factor_name in zip(self.cfg.props, self.factor_names):
            k = max(int(n * prop), 1)
            its = trade_date_data.head(k)["trd_senti"].mean()
            uts = trade_date_data.tail(k)["trd_senti"].mean()
            res[factor_name] = its - uts
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = bgn_date

        # load adj major data as header
        adj_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "oi_instru"],
        )

        # load member
        pos_data = self.load_pos(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=[
                "trade_date", "ts_code", "broker",
                "vol", "long_hld", "long_chg", "short_hld", "short_chg",
                "code_type"
            ]
        )

        cntrct_pos_data = pos_data.query("code_type == 0 and long_hld > 50 and short_hld > 50").dropna(
            axis=0, how="any", subset=["vol", "long_hld", "long_chg", "short_hld", "short_chg"],
        )
        cntrct_pos_data["stat"] = (cntrct_pos_data["long_hld"] + cntrct_pos_data["short_hld"]) / cntrct_pos_data["vol"]
        cntrct_pos_data["abs_chg_sum"] = cntrct_pos_data["long_chg"].abs() + cntrct_pos_data["short_chg"].abs()
        cntrct_pos_data["dlt_chg"] = cntrct_pos_data["long_chg"] - cntrct_pos_data["short_chg"]
        cntrct_pos_data["trd_senti"] = cntrct_pos_data["dlt_chg"] / cntrct_pos_data["abs_chg_sum"]
        cntrct_pos_data = cntrct_pos_data.sort_values(by=["trade_date", "stat"], ascending=[True, False])
        res_df = cntrct_pos_data.groupby(by="trade_date").apply(self.cal_spdweb).reset_index()
        adj_data = pd.merge(left=adj_data, right=res_df, on="trade_date", how="left")
        self.rename_ticker(adj_data)
        factor_data = self.get_factor_data(adj_data, bgn_date)
        return factor_data


class CFactorSIZE(CFactorRaw):
    def __init__(self, cfg: CCfgFactorSIZE, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        size_id = "oi_major"
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", size_id],
        )
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            size_ma = major_data[size_id].rolling(window=win, min_periods=int(win * 0.3)).mean()
            major_data[factor_name] = -(major_data[size_id] / size_ma - 1)
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorHR(CFactorRaw):
    def __init__(self, cfg: CCfgFactorHR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins) + 1, -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "oi_major", "vol_major"],
        )
        major_data["doi"] = (major_data["oi_major"] - major_data["oi_major"].shift(1)).fillna(0)
        hr_id = "hedge_ratio"
        major_data[hr_id] = major_data["doi"] / major_data["vol_major"]
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = major_data[hr_id].rolling(window=win, min_periods=int(win * 0.3)).mean()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorSR(CFactorRaw):
    def __init__(self, cfg: CCfgFactorSR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins) + 1, -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "oi_major", "vol_major"],
        )
        sr_id = "spec_ratio"
        major_data[sr_id] = major_data["vol_major"] / major_data["oi_major"]
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = major_data[sr_id].rolling(window=win, min_periods=int(win * 0.3)).mean()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorLIQUIDITY(CFactorRaw):
    def __init__(self, cfg: CCfgFactorLIQUIDITY, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins) + 1, -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "amount_major"],
        )
        liquidity_id = "liquidity"
        major_data[liquidity_id] = major_data["return_c_major"] * 1e8 / major_data["amount_major"]
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            major_data[factor_name] = major_data[liquidity_id].rolling(window=win, min_periods=int(win * 0.3)).mean()
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorVSTD(CFactorRaw):
    def __init__(self, cfg: CCfgFactorVSTD, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins) + 1, -5)
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "return_c_major", "vol_major"],
        )
        vol_id = "vol_major"
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            mu = major_data[vol_id].rolling(window=win, min_periods=int(win * 0.3)).mean()
            sd = major_data[vol_id].rolling(window=win, min_periods=int(win * 0.3)).std()
            major_data[factor_name] = -(major_data[vol_id] - mu) / sd
        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data


class CFactorAMP(CFactorRaw):
    def __init__(self, cfg: CCfgFactorAMP, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_amp(
            sub_data: pd.DataFrame, x: str, sort_var: str, top_size: int, ascending: bool = False
    ) -> tuple[float, float, float]:
        sorted_data = sub_data.sort_values(by=sort_var, ascending=ascending)
        amp_h = sorted_data.head(top_size)[x].mean()
        amp_l = sorted_data.tail(top_size)[x].mean()
        amp_d = amp_h - amp_l
        return amp_h, amp_l, amp_d

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "highI", "lowI", "closeI"],
        )
        adj_major_data["amp"] = adj_major_data["highI"] / adj_major_data["lowI"] - 1
        adj_major_data["spot"] = adj_major_data["closeI"]

        factor_raw_data = {}
        for win, lbd in ittl.product(self.cfg.wins, self.cfg.lbds):
            top_size = int(win * lbd) + 1
            factor_h, factor_l, factor_d = [
                f"{self.factor_class}{win:03d}T{int(lbd * 10):02d}{_}" for _ in ["H", "L", "D"]
            ]
            r_h_data, r_l_data, r_d_data = {}, {}, {}
            for i, trade_date in enumerate(adj_major_data["trade_date"]):
                if (trade_date < bgn_date) or (trade_date >= stp_date):
                    continue
                sub_data = adj_major_data.iloc[i - win + 1: i + 1]
                rh, rl, rd = self.cal_amp(sub_data=sub_data, x="amp", sort_var="spot", top_size=top_size)
                r_h_data[trade_date], r_l_data[trade_date], r_d_data[trade_date] = rh, rl, rd
            for iter_data, factor in zip([r_h_data, r_l_data, r_d_data], [factor_h, factor_l, factor_d]):
                factor_raw_data[factor] = pd.Series(iter_data)
        factor_raw_df = pd.DataFrame(factor_raw_data)
        input_data = pd.merge(
            left=adj_major_data,
            right=factor_raw_df,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorEXR(CFactorRaw):
    def __init__(self, cfg: CCfgFactorEXR, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def find_extreme_return(tday_minb_data: pd.DataFrame, ret: str, dfts: list[int]) -> pd.Series:
        ret_min, ret_max, ret_median = (
            tday_minb_data[ret].min(),
            tday_minb_data[ret].max(),
            tday_minb_data[ret].median(),
        )
        if (ret_max + ret_min) > (2 * ret_median):
            idx_exr, exr = tday_minb_data[ret].argmax(), -ret_max
        else:
            idx_exr, exr = tday_minb_data[ret].argmin(), -ret_min
        res = {"EXR": exr}
        for d in dfts:
            idx_dxr = idx_exr - d
            dxr = -tday_minb_data[ret].iloc[idx_dxr] if idx_dxr >= 0 else exr
            res[f"DXR{d:02d}"] = dxr
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major"],
        )
        adj_minb_data = self.load_minute_bar(instru, bgn_date=win_start_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        exr_dxr_df = adj_minb_data.groupby(by="trade_date").apply(
            self.find_extreme_return, ret="freq_ret", dfts=self.cfg.dfts  # type:ignore
        )
        factor_win_dfs: list[pd.DataFrame] = []
        for win in self.cfg.wins:
            rename_mapper = {
                **{"EXR": f"EXR{win:03d}"},
                **{f"DXR{d:02d}": f"DXR{win:03d}D{d:02d}" for d in self.cfg.dfts},
            }
            factor_win_data = exr_dxr_df.rolling(window=win).mean()
            factor_win_data = factor_win_data.rename(mapper=rename_mapper, axis=1)
            for d in self.cfg.dfts:
                exr = f"EXR{win:03d}"
                dxr = f"DXR{win:03d}D{d:02d}"
                axr = f"AXR{win:03d}D{d:02d}"
                factor_win_data[axr] = (factor_win_data[exr] + factor_win_data[dxr] * np.sqrt(2)) * 0.5
            factor_win_dfs.append(factor_win_data)
        concat_factor_data = pd.concat(factor_win_dfs, axis=1, ignore_index=False)
        input_data = pd.merge(
            left=adj_major_data,
            right=concat_factor_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorSMT(CFactorRaw):
    def __init__(self, cfg: CCfgFactorSMT, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_smart_idx(data: pd.DataFrame, ret: str, vol: str) -> pd.Series:
        return data[[ret, vol]].apply(lambda z: np.abs(z[ret]) / np.log(z[vol]) * 1e4 if z[vol] > 1 else 0, axis=1)

    @staticmethod
    def cal_smt(sorted_sub_data: pd.DataFrame, lbd: float, prc: str, ret: str) -> tuple[float, float]:
        # total price and ret
        if (tot_amt_sum := sorted_sub_data["amount"].sum()) > 0:
            tot_w = sorted_sub_data["amount"] / tot_amt_sum
            tot_prc = sorted_sub_data[prc] @ tot_w
            tot_ret = sorted_sub_data[ret] @ tot_w
        else:
            return np.nan, np.nan

        # select smart data from total
        volume_threshold = sorted_sub_data["vol"].sum() * lbd
        n = sum(sorted_sub_data["vol"].cumsum() < volume_threshold) + 1
        smt_df = sorted_sub_data.head(n)

        # smart price and ret
        if (smt_amt_sum := smt_df["amount"].sum()) > 0:
            smt_w = smt_df["amount"] / smt_amt_sum
            smt_prc = smt_df[prc] @ smt_w
            smt_ret = smt_df[ret] @ smt_w
            smt_p = ((smt_prc / tot_prc - 1) * 1e4) if tot_prc > 0 else 0
            smt_r = (smt_ret - tot_ret) * 1e4 if not np.isinf(tot_ret) else 0
            return smt_p, smt_r
        else:
            return np.nan, np.nan

    def cal_by_trade_date(self, trade_date_data: pd.DataFrame) -> pd.Series:
        res = {}
        for lbd in self.cfg.lbds:
            p_lbl = f"{self.factor_class}T{int(lbd * 10):02d}P"
            r_lbl = f"{self.factor_class}T{int(lbd * 10):02d}R"
            smt_p, smt_r = self.cal_smt(trade_date_data, lbd=lbd, prc="vwap", ret="freq_ret")
            res[p_lbl], res[r_lbl] = smt_p, smt_r
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        adj_major_data = self.load_preprocess(
            instru, bgn_date=bgn_date, stp_date=stp_date,
            values=["trade_date", "ticker_major"],
        )
        adj_minb_data = self.load_minute_bar(instru, bgn_date=bgn_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)

        # contract multiplier is not considered when calculating "vwap"
        # because a price ratio is considered in the final results, not an absolute value of price is considered
        adj_minb_data["vwap"] = (adj_minb_data["amount"] / adj_minb_data["vol"]).ffill()

        # smart idx
        adj_minb_data["smart_idx"] = self.cal_smart_idx(adj_minb_data, ret="freq_ret", vol="vol")
        adj_minb_data = adj_minb_data.sort_values(by=["trade_date", "smart_idx"], ascending=[True, False])

        concat_factor_data = adj_minb_data.groupby(by="trade_date", group_keys=False).apply(self.cal_by_trade_date)
        input_data = pd.merge(
            left=adj_major_data,
            right=concat_factor_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorRWTC(CFactorRaw):
    def __init__(self, cfg: CCfgFactorRWTC, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    @staticmethod
    def cal_range_weighted_time_center(tday_minb_data: pd.DataFrame, ret: str) -> pd.Series:
        index_reset_df = tday_minb_data.reset_index()
        pos_idx = index_reset_df[ret] > 0
        neg_idx = index_reset_df[ret] < 0
        pos_grp = index_reset_df.loc[pos_idx, ret]
        neg_grp = index_reset_df.loc[neg_idx, ret]
        pos_wgt = pos_grp.abs() / pos_grp.abs().sum()
        neg_wgt = neg_grp.abs() / neg_grp.abs().sum()
        rwtc_u = pos_grp.index @ pos_wgt / len(tday_minb_data)
        rwtc_d = neg_grp.index @ neg_wgt / len(tday_minb_data)
        rwtc_t = rwtc_u - rwtc_d
        rwtc_v = np.abs(rwtc_t)
        return pd.Series({"RWTCU": rwtc_u, "RWTCD": rwtc_d, "RWTCT": rwtc_t, "RWTCV": rwtc_v})

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max(self.cfg.wins), -5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major"],
        )
        adj_minb_data = self.load_minute_bar(instru, bgn_date=win_start_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        rwtc_df = adj_minb_data.groupby(by="trade_date").apply(
            self.cal_range_weighted_time_center, ret="freq_ret"  # type:ignore
        )
        factor_win_dfs: list[pd.DataFrame] = []
        for win in self.cfg.wins:
            rename_mapper = {
                "RWTCU": f"{self.factor_class}{win:03d}U",
                "RWTCD": f"{self.factor_class}{win:03d}D",
                "RWTCT": f"{self.factor_class}{win:03d}T",
                "RWTCV": f"{self.factor_class}{win:03d}V",
            }
            factor_win_data = rwtc_df.rolling(window=win).mean()
            factor_win_data = factor_win_data.rename(mapper=rename_mapper, axis=1)
            factor_win_dfs.append(factor_win_data)
        concat_factor_data = pd.concat(factor_win_dfs, axis=1, ignore_index=False)
        input_data = pd.merge(
            left=adj_major_data,
            right=concat_factor_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorTAILS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorTAILS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_tails_return(self, tday_minb_data: pd.DataFrame, ret: str) -> pd.Series:
        res = {}
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            res[factor_name] = -tday_minb_data[ret].tail(win).mean() * 1e4
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max_win=2, shift=-5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major"],
        )
        adj_minb_data = self.load_minute_bar(instru, bgn_date=win_start_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        tails_data = adj_minb_data.groupby(by="trade_date").apply(self.cal_tails_return, ret="freq_ret")
        input_data = pd.merge(
            left=adj_major_data,
            right=tails_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorHEADS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorHEADS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_heads_return(self, tday_minb_data: pd.DataFrame, ret: str) -> pd.Series:
        res = {}
        for win, factor_name in zip(self.cfg.wins, self.factor_names):
            res[factor_name] = -tday_minb_data[ret].head(win).mean() * 1e4
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max_win=2, shift=-5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major"],
        )
        adj_minb_data = self.load_minute_bar(instru, bgn_date=win_start_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        tails_data = adj_minb_data.groupby(by="trade_date").apply(self.cal_heads_return, ret="freq_ret")
        input_data = pd.merge(
            left=adj_major_data,
            right=tails_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorTOPS(CFactorRaw):
    def __init__(self, cfg: CCfgFactorTOPS, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def cal_tops_return(self, tday_minb_data: pd.DataFrame, ret: str, sort_var: str) -> pd.Series:
        res = {}
        for ratio, factor_name in zip(self.cfg.ratios, self.factor_names):
            sorted_data = tday_minb_data.sort_values(by=sort_var, ascending=False)
            sorted_data["idx"] = sorted_data[sort_var].cumsum()
            vol_sum = sorted_data[sort_var].sum()
            threshold = vol_sum * ratio
            slc_data = sorted_data.query(f"idx <= {threshold}")
            res[factor_name] = slc_data[ret].mean() * 1e4
        return pd.Series(res)

    def cal_factor_by_instru(
            self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar
    ) -> pd.DataFrame:
        win_start_date = calendar.get_start_date(bgn_date, max_win=10, shift=-5)
        adj_major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=["trade_date", "ticker_major", "vol_major"],
        )
        adj_major_data["vol_ma"] = adj_major_data["vol_major"].rolling(window=10).mean()
        adj_major_data["adj_ratio"] = adj_major_data["vol_major"] / adj_major_data["vol_ma"]
        adj_minb_data = self.load_minute_bar(instru, bgn_date=win_start_date, stp_date=stp_date)
        adj_minb_data["freq_ret"] = adj_minb_data["close"] / adj_minb_data["pre_close"] - 1
        adj_minb_data["freq_ret"] = adj_minb_data["freq_ret"].fillna(0)
        tops_data = adj_minb_data.groupby(by="trade_date").apply(self.cal_tops_return, ret="freq_ret", sort_var="vol")
        input_data = pd.merge(
            left=adj_major_data,
            right=tops_data,
            left_on="trade_date",
            right_index=True,
            how="left",
        )
        for factor_name in self.cfg.factor_names:
            input_data[factor_name] = -np.sign(input_data[factor_name]) * input_data["adj_ratio"]
        self.rename_ticker(input_data)
        factor_data = self.get_factor_data(input_data, bgn_date=bgn_date)
        return factor_data


class CFactorTA(CFactorRaw):
    def __init__(self, cfg: CCfgFactorTA, **kwargs):
        self.cfg = cfg
        super().__init__(factor_class=cfg.factor_class, factor_names=cfg.factor_names, **kwargs)

    def __cal_macd(self, close: pd.Series) -> pd.Series:
        fast, slow, diff = self.cfg.macd
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=diff)
        return macdhist

    def __cal_bbands(self, close: pd.Series) -> list[float]:
        timeperiod, up, dn = self.cfg.bbands
        upper, middle, lower = ta.BBANDS(close, timeperiod=timeperiod, nbdevup=up, nbdevdn=dn, matype=0)
        res = []
        for u, m, l, c in zip(upper, middle, lower, close):
            if c >= m:
                res.append((u / c - 1) * 100)
            else:
                res.append((l / c - 1) * 100)
        return res

    def __cal_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        acceleration, maximum = self.cfg.sar
        real = ta.SAR(high, low, acceleration=acceleration, maximum=maximum)
        return (close / real - 1) * 100

    def __cal_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.adx
        return ta.ADX(high, low, close, timeperiod=timeperiod)

    def __cal_bop(self, opn: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        _ = self.cfg.bop
        return ta.BOP(opn, high, low, close)

    def __cal_cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.cci
        return ta.CCI(high, low, close, timeperiod=timeperiod)

    def __cal_cmo(self, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.cmo
        return ta.CMO(close, timeperiod=timeperiod)

    def __cal_rsi(self, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.rsi
        return ta.RSI(close, timeperiod=timeperiod)

    def __cal_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        timeperiod = self.cfg.mfi
        return ta.MFI(high, low, close, volume, timeperiod=timeperiod)

    def __cal_willr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.willr
        return ta.WILLR(high, low, close, timeperiod=timeperiod)

    def __cal_adosc(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        fast, slow = self.cfg.adosc
        adosc = ta.ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)
        vol_ma = volume.rolling(slow).mean()
        return adosc / vol_ma

    def __cal_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        timeperiod = self.cfg.obv
        obv = ta.OBV(close, volume)
        diff_obv = obv - obv.shift(timeperiod)
        vol_ma = volume.rolling(timeperiod).mean()
        return diff_obv / vol_ma

    def __cal_natr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        timeperiod = self.cfg.natr
        return ta.NATR(high, low, close, timeperiod=timeperiod)

    def __cal_ma3(self, close: pd.Series) -> pd.Series:
        fast, mid, slow = self.cfg.ma3
        srs_fast = close.rolling(window=fast).mean()
        srs_mid = close.rolling(window=mid).mean()
        srs_slow = close.rolling(window=slow).mean()
        d0 = srs_fast / srs_mid - 1
        d1 = srs_mid / srs_slow - 1
        a = 1 / 3
        return d0 * a + d1 * (1 - a)

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        win_start_date = "20120104"
        major_data = self.load_preprocess(
            instru, bgn_date=win_start_date, stp_date=stp_date,
            values=[
                "trade_date", "ticker_major",
                "openI", "highI", "lowI", "closeI",
                "vol_major", "amount_major", "oi_major"
            ],
        )
        opn, close = major_data["openI"], major_data["closeI"]
        high, low = major_data["highI"], major_data["lowI"]
        volume, amount = major_data["vol_major"], major_data["amount_major"]

        major_data[self.cfg.name_macd] = self.__cal_macd(close=close)
        major_data[self.cfg.name_bbands] = self.__cal_bbands(close=close)
        major_data[self.cfg.name_sar] = self.__cal_sar(high=high, low=low, close=close)
        major_data[self.cfg.name_adx] = self.__cal_adx(high=high, low=low, close=close)
        major_data[self.cfg.name_bop] = self.__cal_bop(opn=opn, high=high, low=low, close=close)
        major_data[self.cfg.name_cci] = self.__cal_cci(high=high, low=low, close=close)
        major_data[self.cfg.name_cmo] = self.__cal_cmo(close=close)
        major_data[self.cfg.name_rsi] = self.__cal_rsi(close=close)
        major_data[self.cfg.name_mfi] = self.__cal_mfi(high=high, low=low, close=close, volume=volume)
        major_data[self.cfg.name_willr] = self.__cal_willr(high=high, low=low, close=close)
        major_data[self.cfg.name_adosc] = self.__cal_adosc(high=high, low=low, close=close, volume=volume)
        major_data[self.cfg.name_obv] = self.__cal_obv(close=close, volume=volume)
        major_data[self.cfg.name_natr] = self.__cal_natr(high=high, low=low, close=close)
        major_data[self.cfg.name_ma3] = self.__cal_ma3(close=close)

        self.rename_ticker(major_data)
        factor_data = self.get_factor_data(major_data, bgn_date)
        return factor_data
