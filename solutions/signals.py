import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import Progress, track
from husfort.qutility import check_and_makedirs, error_handler
from husfort.qsqlite import CMgrSqlDb, CDbStruct
from husfort.qcalendar import CCalendar
from solutions.shared import gen_sig_db, gen_fac_agg_db, gen_opt_wgt_db, gen_prdct_db
from typedef import CFactor, TFactors, TFactorNames, CSimArgs, TRetPrc, CTestMdl


class _CSignal:
    def __init__(self, signal_save_dir: str, signal_id: str):
        self.signal_save_dir = signal_save_dir
        self.signal_id = signal_id

    def save(self, new_data: pd.DataFrame, calendar: CCalendar):
        db_struct_sig = gen_sig_db(self.signal_save_dir, self.signal_id)
        check_and_makedirs(db_struct_sig.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_sig.db_save_dir,
            db_name=db_struct_sig.db_name,
            table=db_struct_sig.table,
            mode="a",
        )
        if sqldb.check_continuity(new_data["trade_date"].iloc[0], calendar) == 0:
            sqldb.update(new_data[db_struct_sig.table.vars.names])
        return 0

    def read(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_sig = gen_sig_db(self.signal_save_dir, self.signal_id)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_sig.db_save_dir,
            db_name=db_struct_sig.db_name,
            table=db_struct_sig.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date, stp_date)
        return data

    def load_input(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        raise NotImplementedError

    def core(self, input_data: pd.DataFrame, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        raise NotImplementedError

    def main(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        input_data = self.load_input(bgn_date, stp_date, calendar)
        new_data = self.core(input_data, bgn_date, stp_date, calendar)
        self.save(new_data=new_data, calendar=calendar)
        return 0

    @staticmethod
    def map_factor_to_signal(data: pd.DataFrame) -> pd.DataFrame:
        n = len(data)
        k = int(n * 0.4)
        data["weight"] = [1] * k + [0] * (n - 2 * k) + [-1] * k
        if (abs_sum := data["weight"].abs().sum()) > 0:
            data["weight"] = data["weight"] / abs_sum
        return data[["trade_date", "instrument", "weight"]]

    @staticmethod
    def norm_scale(data: pd.DataFrame) -> pd.DataFrame:
        if (abs_sum := data["weight"].abs().sum()) > 0:
            data["weight"] = data["weight"] / abs_sum
        return data[["trade_date", "instrument", "weight"]]

    @staticmethod
    def moving_average_signal(
            signal_data: pd.DataFrame, bgn_date: str, decay_rate: float, decay_win: int
    ) -> pd.DataFrame:
        """

        :param signal_data: pd.Dataframe with columns = ["trade_date", "instrument", "weight"]
        :param bgn_date:
        :param decay_rate:
        :param decay_win:
        :return:
        """
        rou = np.power(decay_rate, 1 / decay_win)  # rou **decay_win = decay_rate
        w = np.power(rou, np.arange(decay_win, 0, -1))
        w = w / w.sum()

        pivot_data = pd.pivot_table(
            data=signal_data,
            index=["trade_date"],
            columns=["instrument"],
            values=["weight"],
        )
        instru_ma_data = pivot_data.fillna(0).rolling(window=decay_win).apply(lambda z: z @ w)
        truncated_data = instru_ma_data.query(f"trade_date >= '{bgn_date}'")
        normalize_data = truncated_data.div(truncated_data.abs().sum(axis=1), axis=0).fillna(0)
        stack_data = normalize_data.stack(future_stack=True).reset_index()
        return stack_data[["trade_date", "instrument", "weight"]]


"""
-------------------------------------------
--- signals from aggregated raw factors ---
-------------------------------------------
"""


class CSignalFromFactor(_CSignal):
    def __init__(self, factor: CFactor, factor_save_root_dir: str, signal_save_dir: str,
                 decay_rate: float, decay_win: int):
        self.factor = factor
        self.factor_save_root_dir = factor_save_root_dir
        self.decay_rate, self.decay_win = decay_rate, decay_win
        signal_id = factor.factor_name
        super().__init__(signal_save_dir=signal_save_dir, signal_id=signal_id)

    def load_input(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_bgn_date = calendar.get_next_date(bgn_date, -self.decay_win + 1)
        db_struct_fac = gen_fac_agg_db(
            db_save_root_dir=self.factor_save_root_dir,
            factor_class=self.factor.factor_class,
            factor_names=TFactorNames([self.factor.factor_name]),
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_fac.db_save_dir,
            db_name=db_struct_fac.db_name,
            table=db_struct_fac.table,
            mode="r",
        )
        data = sqldb.read_by_range(
            bgn_date=base_bgn_date, stp_date=stp_date,
            value_columns=["trade_date", "instrument", self.factor.factor_name],
        )
        return data

    def core(self, input_data: pd.DataFrame, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        sorted_data = input_data.sort_values(
            by=["trade_date", self.factor.factor_name, "instrument"], ascending=[True, False, True]
        )
        grouped_data = sorted_data.groupby(by=["trade_date"], group_keys=False)
        signal_data = grouped_data.apply(self.map_factor_to_signal)
        signal_data_ma = self.moving_average_signal(
            signal_data, bgn_date=bgn_date, decay_rate=self.decay_rate, decay_win=self.decay_win)
        return signal_data_ma


def process_for_signal_from_factor_agg(
        factor: CFactor, factor_save_root_dir: str,
        decay_rate: float, decay_win: int,
        signal_save_dir: str,
        bgn_date: str, stp_date: str, calendar: CCalendar,
):
    signal = CSignalFromFactor(
        factor, factor_save_root_dir, signal_save_dir, decay_rate=decay_rate, decay_win=decay_win)
    signal.main(bgn_date, stp_date, calendar)
    return 0


def main_signals_from_factor_agg(
        factors: TFactors,
        factor_save_root_dir: str,
        decay_rate: float,
        decay_win: int,
        signal_save_dir: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Translating factors to signals ..."
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(factors))
            with mp.get_context("spawn").Pool(processes) as pool:
                for factor in factors:
                    pool.apply_async(
                        process_for_signal_from_factor_agg,
                        kwds={
                            "factor": factor,
                            "factor_save_root_dir": factor_save_root_dir,
                            "decay_rate": decay_rate,
                            "decay_win": decay_win,
                            "signal_save_dir": signal_save_dir,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for factor in track(factors, description=desc):
            process_for_signal_from_factor_agg(
                factor=factor,
                factor_save_root_dir=factor_save_root_dir,
                decay_rate=decay_rate,
                decay_win=decay_win,
                signal_save_dir=signal_save_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0


"""
---------------------------------------
--- signals from model optimization ---
---------------------------------------
"""


class CSignalFromOpt(_CSignal):
    def __init__(
            self, group_id: TRetPrc, sim_args_list: list[CSimArgs],
            input_sig_dir: str,
            input_opt_dir: str,
            signal_save_dir: str,
            db_struct_avlb: CDbStruct,
    ):
        signal_id = group_id
        self.db_struct_avlb = db_struct_avlb
        self.input_signal_ids: list[str] = [sim_args.sim_id for sim_args in sim_args_list]
        self.input_sig_dir = input_sig_dir
        self.input_opt_dir = input_opt_dir
        super().__init__(signal_save_dir=signal_save_dir, signal_id=signal_id)

    @property
    def underlying_assets_names(self) -> list[str]:
        return [input_signal_id.split(".")[0] for input_signal_id in self.input_signal_ids]

    def load_available_data(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_avlb.db_save_dir,
            db_name=self.db_struct_avlb.db_name,
            table=self.db_struct_avlb.table,
            mode="r",
        )
        return sqldb.read_by_range(bgn_date, stp_date)

    def load_opt(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_opt = gen_opt_wgt_db(
            db_save_dir=self.input_opt_dir,
            save_id=self.signal_id,
            underlying_assets_names=self.underlying_assets_names,
        )
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_opt.db_save_dir,
            db_name=db_struct_opt.db_name,
            table=db_struct_opt.table,
            mode="r",
        )
        data = sqldb.read_by_range(bgn_date=bgn_date, stp_date=stp_date)
        return data.set_index("trade_date")

    def load_input(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        input_data = {}
        for input_signal_id in self.input_signal_ids:
            signal_id = ".".join(input_signal_id.split(".")[:-1])
            unique_id = input_signal_id.split(".")[0]
            db_struct_sig = gen_sig_db(db_save_dir=self.input_sig_dir, signal_id=signal_id)
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_sig.db_save_dir,
                db_name=db_struct_sig.db_name,
                table=db_struct_sig.table,
                mode="r",
            )
            data = sqldb.read_by_range(bgn_date=bgn_date, stp_date=stp_date)
            input_data[unique_id] = data.set_index(["trade_date", "instrument"])["weight"]
        return pd.DataFrame(input_data)

    @staticmethod
    def apply_opt(sorted_data: pd.DataFrame, opt_data: pd.DataFrame) -> pd.DataFrame:
        res = []
        for trade_date, trade_date_data in sorted_data.groupby(by="trade_date"):
            srs = trade_date_data @ opt_data.loc[trade_date]
            if (abs_sum := srs.abs().sum()) > 0:
                srs = srs / abs_sum
            res.append(srs)
        optimized_data = pd.concat(res).reset_index().rename(columns={0: "weight"})
        return optimized_data

    @staticmethod
    def apply_risk_control(sig_data: pd.DataFrame, avlb_data: pd.DataFrame) -> pd.DataFrame:
        def __normalize(s: pd.Series):
            size = len(s)
            u = 3 / size
            if any(idx := s.abs().ge(u)):  # type:ignore
                s[idx] = np.sign(s[idx]) * u
            s = s / s.abs().sum()
            return s

        avlb_sig_data = pd.merge(
            left=avlb_data[["trade_date", "instrument", "sectorL1"]],
            right=sig_data,
            on=["trade_date", "instrument"],
            how="left",
        )
        adj_wgt_data = avlb_sig_data.groupby(by=["trade_date"], group_keys=False)["weight"].apply(__normalize)
        avlb_sig_data["weight_adj"] = adj_wgt_data
        risk_ctl_data = avlb_sig_data[["trade_date", "instrument", "weight_adj"]]
        return risk_ctl_data.rename(columns={"weight_adj": "weight"})

    def core(self, input_data: pd.DataFrame, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        sorted_data = input_data.sort_values(by="trade_date", ascending=True).fillna(0)
        opt_data = self.load_opt(bgn_date, stp_date)
        signal_data = self.apply_opt(sorted_data, opt_data)
        avlb_data = self.load_available_data(bgn_date, stp_date)
        adj_data = self.apply_risk_control(signal_data, avlb_data)
        return adj_data


def process_for_signal_from_opt(
        group_id: TRetPrc,
        sim_args_list: list[CSimArgs],
        input_sig_dir: str,
        input_opt_dir: str,
        signal_save_dir: str,
        db_struct_avlb: CDbStruct,
        bgn_date: str, stp_date: str, calendar: CCalendar,
):
    signal = CSignalFromOpt(
        group_id=group_id, sim_args_list=sim_args_list, input_sig_dir=input_sig_dir,
        input_opt_dir=input_opt_dir, signal_save_dir=signal_save_dir, db_struct_avlb=db_struct_avlb,
    )
    signal.main(bgn_date, stp_date, calendar)
    return 0


def main_signals_from_opt(
        grouped_sim_args: dict[TRetPrc, list[CSimArgs]],
        input_sig_dir: str,
        input_opt_dir: str,
        signal_save_dir: str,
        db_struct_avlb: CDbStruct,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Translating optimized models to signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(grouped_sim_args))
            with mp.get_context("spawn").Pool(processes) as pool:
                for group_id, sim_args_list in grouped_sim_args.items():
                    pool.apply_async(
                        process_for_signal_from_opt,
                        kwds={
                            "group_id": group_id,
                            "sim_args_list": sim_args_list,
                            "input_sig_dir": input_sig_dir,
                            "input_opt_dir": input_opt_dir,
                            "signal_save_dir": signal_save_dir,
                            "db_struct_avlb": db_struct_avlb,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for group_id, sim_args_list in track(grouped_sim_args.items(), description=desc):
            process_for_signal_from_opt(
                group_id=group_id,
                sim_args_list=sim_args_list,
                input_sig_dir=input_sig_dir,
                input_opt_dir=input_opt_dir,
                signal_save_dir=signal_save_dir,
                db_struct_avlb=db_struct_avlb,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0


"""
-------------------------------------
--- signals from model prediction ---
-------------------------------------
"""


class CSignalFromMdlPrd(_CSignal):
    def __init__(self, test: CTestMdl, mclrn_prd_dir: str, signal_save_dir: str, decay_rate: float, decay_win: int):
        self.test = test
        self.mclrn_prd_dir = mclrn_prd_dir
        self.decay_rate, self.decay_win = decay_rate, decay_win
        signal_id = test.save_tag_mdl
        super().__init__(signal_save_dir=signal_save_dir, signal_id=signal_id)

    def load_input(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_bgn_date = calendar.get_next_date(bgn_date, -self.decay_win + 1)
        db_struct_prd = gen_prdct_db(db_save_root_dir=self.mclrn_prd_dir, test=self.test)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_prd.db_save_dir,
            db_name=db_struct_prd.db_name,
            table=db_struct_prd.table,
            mode="r",
        )
        data = sqldb.read_by_range(
            bgn_date=base_bgn_date, stp_date=stp_date,
            value_columns=["trade_date", "instrument", self.test.ret.ret_name],
        )
        return data

    def core(self, input_data: pd.DataFrame, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        sorted_data = input_data.sort_values(
            by=["trade_date", self.test.ret.ret_name, "instrument"], ascending=[True, False, True]
        )
        grouped_data = sorted_data.groupby(by=["trade_date"], group_keys=False)
        signal_data = grouped_data.apply(self.map_factor_to_signal)
        signal_data_ma = self.moving_average_signal(
            signal_data, bgn_date=bgn_date, decay_rate=self.decay_rate, decay_win=self.decay_win)
        return signal_data_ma


def process_for_signal_from_mdl_prd(
        test: CTestMdl, mclrn_prd_dir: str, signal_save_dir: str,
        decay_rate: float, decay_win: int,
        bgn_date: str, stp_date: str, calendar: CCalendar,
):
    signal = CSignalFromMdlPrd(test=test, mclrn_prd_dir=mclrn_prd_dir, signal_save_dir=signal_save_dir,
                               decay_rate=decay_rate, decay_win=decay_win)
    signal.main(bgn_date, stp_date, calendar)
    return 0


def main_signals_from_mdl_prd(
        tests: list[CTestMdl],
        mclrn_prd_dir: str,
        signal_save_dir: str,
        decay_rate: float, decay_win: int,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Translating machine learning model predictions to signals"
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(tests))
            with mp.get_context("spawn").Pool(processes) as pool:
                for test in tests:
                    pool.apply_async(
                        process_for_signal_from_mdl_prd,
                        kwds={
                            "test": test,
                            "mclrn_prd_dir": mclrn_prd_dir,
                            "signal_save_dir": signal_save_dir,
                            "decay_rate": decay_rate,
                            "decay_win": decay_win,
                            "bgn_date": bgn_date,
                            "stp_date": stp_date,
                            "calendar": calendar,
                        },
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                pool.close()
                pool.join()
    else:
        for test in track(tests, description=desc):
            process_for_signal_from_mdl_prd(
                test=test,
                mclrn_prd_dir=mclrn_prd_dir,
                signal_save_dir=signal_save_dir,
                decay_rate=decay_rate,
                decay_win=decay_win,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    return 0
