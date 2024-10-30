import numpy as np
import pandas as pd
import multiprocessing as mp
from rich.progress import track, Progress
from husfort.qutility import SFY, error_handler, check_and_makedirs
from husfort.qsqlite import CDbStruct, CMgrSqlDb
from husfort.qcalendar import CCalendar
from typedef import TFactorClass, TFactorNames, TUniverse, TFactorName
from solutions.shared import gen_fac_raw_db, gen_fac_neu_db, neutralize_by_date


class CFactorGeneric:
    def __init__(self, factor_class: TFactorClass, factor_names: TFactorNames, save_by_instru_dir: str):
        self.factor_class = factor_class
        self.factor_names = factor_names
        self.save_by_instru_dir: str = save_by_instru_dir

    def load_by_instru(self, instru: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        db_struct_instru = gen_fac_raw_db(instru, self.save_by_instru_dir, self.factor_class, self.factor_names)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="r",
        )
        factor_data = sqldb.read_by_range(bgn_date, stp_date)
        factor_data[self.factor_names] = factor_data[self.factor_names].astype(np.float64).fillna(np.nan)
        return factor_data

    def save_raw_by_instru(self, factor_data: pd.DataFrame, instru: str, calendar: CCalendar):
        db_struct_instru = gen_fac_raw_db(instru, self.save_by_instru_dir, self.factor_class, self.factor_names)
        check_and_makedirs(db_struct_instru.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_instru.db_save_dir,
            db_name=db_struct_instru.db_name,
            table=db_struct_instru.table,
            mode="a",
        )
        if sqldb.check_continuity(factor_data["trade_date"].iloc[0], calendar) == 0:
            sqldb.update(factor_data[db_struct_instru.table.vars.names])
        return 0

    def save_neu_by_class(self, factor_data: pd.DataFrame, calendar: CCalendar):
        db_struct_class = gen_fac_neu_db(self.save_by_instru_dir, self.factor_class, self.factor_names)
        check_and_makedirs(db_struct_class.db_save_dir)
        sqldb = CMgrSqlDb(
            db_save_dir=db_struct_class.db_save_dir,
            db_name=db_struct_class.db_name,
            table=db_struct_class.table,
            mode="a",
        )
        if sqldb.check_continuity(factor_data["trade_date"].iloc[0], calendar) == 0:
            sqldb.update(factor_data[db_struct_class.table.vars.names])
        return 0

    def get_factor_data(self, input_data: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        input_data = input_data.query(f"trade_date >= '{bgn_date}'")
        factor_data = input_data[["trade_date", "ticker"] + self.factor_names]
        return factor_data

    @staticmethod
    def rename_ticker(data: pd.DataFrame, old_name: str = "ticker_major") -> None:
        data.rename(columns={old_name: "ticker"}, inplace=True)


class CFactorRaw(CFactorGeneric):
    def __init__(
            self,
            factor_class: TFactorClass,
            factor_names: TFactorNames,
            factors_by_instru_dir: str,
            universe: TUniverse,
            db_struct_preprocess: CDbStruct | None = None,
            db_struct_minute_bar: CDbStruct | None = None,
            db_struct_pos: CDbStruct | None = None,
            db_struct_forex: CDbStruct | None = None,
            db_struct_macro: CDbStruct | None = None,
            db_struct_mkt: CDbStruct | None = None,
    ):
        super().__init__(factor_class, factor_names, save_by_instru_dir=factors_by_instru_dir)
        self.universe = universe
        self.db_struct_preprocess = db_struct_preprocess
        self.db_struct_minute_bar = db_struct_minute_bar
        self.db_struct_pos = db_struct_pos
        self.db_struct_forex = db_struct_forex
        self.db_struct_macro = db_struct_macro
        self.db_struct_mkt = db_struct_mkt

    def load_preprocess(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_preprocess is not None:
            db_struct_instru = self.db_struct_preprocess.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_preprocess' must be provided")

    def load_minute_bar(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_minute_bar is not None:
            db_struct_instru = self.db_struct_minute_bar.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_minute_bar' must be provided")

    def load_pos(self, instru: str, bgn_date: str, stp_date: str, values: list[str] = None) -> pd.DataFrame:
        if self.db_struct_pos is not None:
            db_struct_instru = self.db_struct_pos.copy_to_another(another_db_name=f"{instru}.db")
            sqldb = CMgrSqlDb(
                db_save_dir=db_struct_instru.db_save_dir,
                db_name=db_struct_instru.db_name,
                table=db_struct_instru.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date, value_columns=values)
        else:
            raise ValueError("Argument 'db_struct_pos' must be provided")

    def load_forex(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_forex is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_forex.db_save_dir,
                db_name=self.db_struct_forex.db_name,
                table=self.db_struct_forex.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_forex' must be provided")

    def load_macro(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_macro is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_macro.db_save_dir,
                db_name=self.db_struct_macro.db_name,
                table=self.db_struct_macro.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_macro' must be provided")

    def load_mkt(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        if self.db_struct_mkt is not None:
            sqldb = CMgrSqlDb(
                db_save_dir=self.db_struct_mkt.db_save_dir,
                db_name=self.db_struct_mkt.db_name,
                table=self.db_struct_mkt.table,
                mode="r",
            )
            return sqldb.read_by_range(bgn_date, stp_date)
        else:
            raise ValueError("Argument 'db_struct_mkt' must be provided")

    def cal_factor_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        """
        This function is to be realized by specific factors

        :return : a pd.DataFrame with first 3 columns must be = ["tp", "trade_date", "ticker"]
                  then followed by factor names
        """
        raise NotImplementedError

    def process_by_instru(self, instru: str, bgn_date: str, stp_date: str, calendar: CCalendar):
        factor_data = self.cal_factor_by_instru(instru, bgn_date, stp_date, calendar)
        self.save_raw_by_instru(factor_data, instru, calendar)
        return 0

    def main_raw(self, bgn_date: str, stp_date: str, calendar: CCalendar, call_multiprocess: bool, processes: int):
        description = f"Calculating factor {SFY(self.factor_class)}"
        if call_multiprocess:
            with Progress() as pb:
                main_task = pb.add_task(description, total=len(self.universe))
                with mp.get_context("spawn").Pool(processes) as pool:
                    for instru in self.universe:
                        pool.apply_async(
                            self.process_by_instru,
                            args=(instru, bgn_date, stp_date, calendar),
                            callback=lambda _: pb.update(main_task, advance=1),
                            error_callback=error_handler,
                        )
                    pool.close()
                    pool.join()
        else:
            for instru in track(self.universe, description=description):
                self.process_by_instru(instru, bgn_date, stp_date, calendar)
        return 0


# --------------------------------------------
# -------------- Neutralization --------------
# --------------------------------------------

class CFactorNeu(CFactorGeneric):
    def __init__(
            self,
            ref_factor: CFactorGeneric,
            universe: TUniverse,
            db_struct_preprocess: CDbStruct,
            db_struct_avlb: CDbStruct,
            neutral_by_instru_dir: str,
    ):
        self.ref_factor: CFactorGeneric = ref_factor
        self.universe = universe
        self.db_struct_preprocess = db_struct_preprocess
        self.db_struct_avlb = db_struct_avlb
        super().__init__(
            factor_class=self.ref_factor.factor_class,
            factor_names=TFactorNames([TFactorName(z.replace("RAW", "NEU")) for z in self.ref_factor.factor_names]),
            save_by_instru_dir=neutral_by_instru_dir,
        )

    def load_ref_factor(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        ref_dfs: list[pd.DataFrame] = []
        for instru in self.universe:
            df = self.ref_factor.load_by_instru(instru, bgn_date, stp_date)
            df["instrument"] = instru
            ref_dfs.append(df)
        res = pd.concat(ref_dfs, axis=0, ignore_index=False)
        res = res.sort_values(by="trade_date", ascending=True)
        res = res[["trade_date", "instrument"] + self.ref_factor.factor_names]
        return res

    def load_available(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_avlb.db_save_dir,
            db_name=self.db_struct_avlb.db_name,
            table=self.db_struct_avlb.table,
            mode="r",
        )
        df = sqldb.read_by_range(bgn_date, stp_date)
        df = df[["trade_date", "instrument", "sectorL1"]]
        return df

    def main_neu(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        ref_factor_data = self.load_ref_factor(bgn_date, stp_date)
        available_data = self.load_available(bgn_date, stp_date)
        net_ref_factor_data = pd.merge(
            left=available_data,
            right=ref_factor_data,
            on=["trade_date", "instrument"],
            how="left",
        ).sort_values(by=["trade_date", "sectorL1"])
        neu_factor_data = neutralize_by_date(
            net_ref_factor_data, old_names=self.ref_factor.factor_names, new_names=self.factor_names,
            date_name="trade_date", sec_name="sectorL1", instru_name="instrument",
        )
        neu_factor_data[self.factor_names] = neu_factor_data[self.factor_names].fillna(0)
        self.save_neu_by_class(factor_data=neu_factor_data, calendar=calendar)
        return 0
