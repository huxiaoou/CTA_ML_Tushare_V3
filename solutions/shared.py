import os
import pandas as pd
import scipy.stats as sps
import itertools as ittl
from rich.progress import Progress
from husfort.qsqlite import CDbStruct, CSqlTable, CSqlVar
from typedef import TFactorClass, TFactorName, TFactorNames, TFactors, CSimArgs, TRets, TUniqueId, TGroupId, TRetPrc
from typedef import TSimGrpIdByFacAgg, TSimGrpIdByFacGrp
from typedef import CTestMdl, CRet, CModel, TFactorGroups


# ---------------------------------------
# ------ algorithm: neutralization ------
# ---------------------------------------

def neutralize_by_date(
        raw_data: pd.DataFrame,
        old_names: list[str],
        new_names: list[str],
        date_name: str = "trade_date",
        sec_name: str = "sectorL1",
        instru_name: str = "instrument",
) -> pd.DataFrame:
    """

    :param raw_data: a dataframe with columns = [date_name, instru_name, sec_name] + old_names
    :param old_names:
    :param new_names:
    :param date_name:
    :param sec_name:
    :param instru_name:
    :return: a dataframe, old_names are normalized, and renamed as new_names
             columns = [date_name, instru_name, sec_name] + new_names
    """

    with Progress() as pb:
        task = pb.add_task(description="Neutralizing", total=3)

        # --- get instrument rank for each day
        rank_data = (
            raw_data[[date_name] + old_names]
            .groupby(by=date_name, group_keys=False)[old_names]
            .apply(lambda z: z.rank() / (z.count() + 1))
        )
        pb.update(task, advance=1)

        # --- map rank to random variable with normal distribution
        norm_rv_data = rank_data.apply(sps.norm.ppf)
        norm_data = pd.merge(
            left=raw_data[[date_name, instru_name, sec_name]],
            right=norm_rv_data,
            how="inner",
            left_index=True, right_index=True,
        )
        pb.update(task, advance=1)

        # --- neutralize for each sector and day
        neu_data = (
            norm_data[[date_name, sec_name] + old_names]
            .groupby(by=[date_name, sec_name], group_keys=False)[old_names]
            .apply(lambda z: z - z.mean())
        )
        rename_mapper = {o: n for o, n in zip(old_names, new_names)}
        res_data = pd.merge(
            left=raw_data[[date_name, instru_name, sec_name]],
            right=neu_data,
            how="inner",
            left_index=True, right_index=True
        ).rename(columns=rename_mapper)
        pb.update(task, advance=1)

        # --- reformat
        res_data = res_data[[date_name, instru_name, sec_name] + new_names]
    return res_data


# ----------------------------------------
# ------ sqlite3 database structure ------
# ----------------------------------------

def gen_tst_ret_raw_db(instru: str, db_save_root_dir: str, save_id: str, rets: list[str]) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, save_id),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="test_return",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(ret, "REAL") for ret in rets],
        )
    )


def gen_tst_ret_agg_db(db_save_root_dir: str, save_id: str, rets: list[str]) -> CDbStruct:
    """

    :param db_save_root_dir:
    :param save_id: like "001L1RAW"
    :param rets: like ["Cls010L1RAW", "Opn010L1RAW"]
    :return:
    """

    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name="test_return",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(ret, "REAL") for ret in rets],
        )
    )


def gen_fac_raw_db(
        instru: str, db_save_root_dir: str, factor_class: TFactorClass, factor_names: TFactorNames,
) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, factor_class),
        db_name=f"{instru}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar("ticker", "TEXT")] + [CSqlVar(fn, "REAL") for fn in factor_names],
        )
    )


def gen_fac_agg_db(
        db_save_root_dir: str, factor_class: TFactorClass, factor_names: TFactorNames,
) -> CDbStruct:
    return CDbStruct(
        db_save_dir=os.path.join(db_save_root_dir, factor_class),
        db_name=f"{factor_class}.db",
        table=CSqlTable(
            name="factor",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(fn, "REAL") for fn in factor_names],
        )
    )


def gen_sig_db(db_save_dir: str, signal_id: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_dir,
        db_name=f"{signal_id}.db",
        table=CSqlTable(
            name="signal",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar("weight", "REAL")],
        )
    )


def gen_nav_db(db_save_dir: str, save_id: str) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name="nav",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[
                CSqlVar("raw_ret", "REAL"),
                CSqlVar("dlt_wgt", "REAL"),
                CSqlVar("cost", "REAL"),
                CSqlVar("net_ret", "REAL"),
                CSqlVar("nav", "REAL"),
            ],
        )
    )


def gen_prdct_db(db_save_root_dir: str, test: CTestMdl) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_root_dir,
        db_name=f"{test.save_tag_mdl}.db",
        table=CSqlTable(
            name="prediction",
            primary_keys=[CSqlVar("trade_date", "TEXT"), CSqlVar("instrument", "TEXT")],
            value_columns=[CSqlVar(test.ret.ret_name, "REAL")],
        )
    )


def gen_opt_wgt_db(db_save_dir: str, save_id: str, underlying_assets_names: list[str]) -> CDbStruct:
    return CDbStruct(
        db_save_dir=db_save_dir,
        db_name=f"{save_id}.db",
        table=CSqlTable(
            name="weights",
            primary_keys=[CSqlVar("trade_date", "TEXT")],
            value_columns=[CSqlVar(_, "REAL") for _ in underlying_assets_names],
        )
    )


# -----------------------------------------
# ------ arguments about simulations ------
# -----------------------------------------

def get_sim_args_fac(
        factors: TFactors, maws: list[int], rets: TRets, signals_dir: str, ret_dir: str, cost: float,
) -> list[CSimArgs]:
    res: list[CSimArgs] = []
    for factor, maw, ret in ittl.product(factors, maws, rets):
        signal_id = f"{factor.factor_name}.MA{maw:02d}"
        ret_names = [ret.ret_name]
        sim_args = CSimArgs(
            sim_id=f"{signal_id}.{ret.ret_name}",
            tgt_ret=ret,
            db_struct_sig=gen_sig_db(db_save_dir=signals_dir, signal_id=signal_id),
            db_struct_ret=gen_tst_ret_agg_db(db_save_root_dir=ret_dir, save_id=ret.save_id, rets=ret_names),
            cost=cost,
        )
        res.append(sim_args)
    return res


def group_sim_args_by_factor_class(
        sim_args_list: list[CSimArgs], mapper_name_to_class: dict[TFactorName, TFactorClass],
) -> dict[TSimGrpIdByFacAgg, list[CSimArgs]]:
    res: dict[TSimGrpIdByFacAgg, list[CSimArgs]] = {}
    for sim_args in sim_args_list:
        factor_name, maw, ret_name = sim_args.sim_id.split(".")
        factor_class = mapper_name_to_class[TFactorName(factor_name)]
        ret_prc = ret_name[0:3]
        key = TSimGrpIdByFacAgg((factor_class, ret_prc, maw))
        if key not in res:
            res[key] = []
        res[key].append(sim_args)
    return res

# -----------------------------------------
# ------ arguments about simulations ------
# -----------------------------------------
