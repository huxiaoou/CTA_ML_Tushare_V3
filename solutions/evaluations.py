import os
import multiprocessing as mp
import pandas as pd
from rich.progress import track, Progress
from husfort.qutility import error_handler, check_and_makedirs
from husfort.qevaluation import CNAV
from husfort.qsqlite import CMgrSqlDb, CDbStruct
from husfort.qplot import CPlotLines
from solutions.shared import gen_nav_db
from typedef import CSimArgs, TSimGrpIdByFacAgg, TRetPrc


class CEvl:
    def __init__(self, db_struct_nav: CDbStruct):
        self.db_struct_nav = db_struct_nav
        self.indicators = ("hpr", "retMean", "retStd", "retAnnual", "volAnnual", "sharpe", "calmar", "mdd")

    def load(self, bgn_date: str, stp_date: str) -> pd.DataFrame:
        sqldb = CMgrSqlDb(
            db_save_dir=self.db_struct_nav.db_save_dir,
            db_name=self.db_struct_nav.db_name,
            table=self.db_struct_nav.table,
            mode="r"
        )
        nav_data = sqldb.read_by_range(bgn_date, stp_date)
        return nav_data

    def add_arguments(self, res: dict):
        raise NotImplementedError

    def get_ret(self, bgn_date: str, stp_date: str) -> pd.Series:
        """

        :param bgn_date:
        :param stp_date:
        :return: a pd.Series, with string index
        """
        nav_data = self.load(bgn_date, stp_date)
        ret_srs = nav_data.set_index("trade_date")["net_ret"]
        return ret_srs

    def main(self, bgn_date: str, stp_date: str) -> dict:
        ret_srs = self.get_ret(bgn_date, stp_date)
        nav = CNAV(ret_srs, input_type="RET")
        nav.cal_all_indicators()
        res = nav.to_dict()
        res = {k: res[k] for k in self.indicators}
        self.add_arguments(res)
        return res


class CEvlFrmSim(CEvl):
    def __init__(self, sim_args: CSimArgs, sim_save_dir: str):
        self.sim_args = sim_args
        db_struct_nav = gen_nav_db(db_save_dir=sim_save_dir, save_id=sim_args.sim_id)
        super().__init__(db_struct_nav)


class CEvlFacAgg(CEvlFrmSim):
    """
    --- evaluations for neutralized factors ---
    """

    def add_arguments(self, res: dict):
        factor_name, ret_name = self.sim_args.sim_id.split(".")
        other_arguments = {
            "factor_name": factor_name,
            "ret_name": ret_name,
        }
        res.update(other_arguments)
        return 0


class CEvlFacOpt(CEvlFrmSim):
    """
    --- evaluations for price type ---
    """

    def add_arguments(self, res: dict):
        ret_prc, tgt_ret = self.sim_args.sim_id.split(".")
        other_arguments = {
            "ret_prc": ret_prc,
            "tgt_ret": tgt_ret,
        }
        res.update(other_arguments)
        return 0


def process_for_evl_frm_sim(
        sim_type: str,
        sim_args: CSimArgs,
        sim_save_dir: str,
        bgn_date: str,
        stp_date: str,
) -> dict:
    if sim_type == "facAgg":
        s = CEvlFacAgg(sim_args, sim_save_dir=sim_save_dir)
    elif sim_type == "facOpt":
        s = CEvlFacOpt(sim_args, sim_save_dir=sim_save_dir)
    else:
        raise ValueError(f"sim type = {sim_type} is illegal")
    return s.main(bgn_date, stp_date)


def main_evl_sims(
        sim_type: str,
        sim_args_list: list[CSimArgs],
        sim_save_dir: str,
        evl_save_dir: str,
        evl_save_file: str,
        header_vars: list[str],
        sort_vars: list[str],
        bgn_date: str,
        stp_date: str,
        call_multiprocess: bool,
        processes: int,
):
    desc = "Calculating evaluations for simulations"
    evl_sims: list[dict] = []
    if call_multiprocess:
        with Progress() as pb:
            main_task = pb.add_task(description=desc, total=len(sim_args_list))
            with mp.get_context("spawn").Pool(processes=processes) as pool:
                jobs = []
                for sim_args in sim_args_list:
                    job = pool.apply_async(
                        process_for_evl_frm_sim,
                        args=(sim_type, sim_args, sim_save_dir, bgn_date, stp_date),
                        callback=lambda _: pb.update(main_task, advance=1),
                        error_callback=error_handler,
                    )
                    jobs.append(job)
                pool.close()
                pool.join()
            evl_sims = [job.get() for job in jobs]
    else:
        for sim_args in track(sim_args_list, description=desc):
            evl = process_for_evl_frm_sim(sim_type, sim_args, sim_save_dir, bgn_date, stp_date)
            evl_sims.append(evl)

    evl_data = pd.DataFrame(evl_sims)
    evl_data["sharpe+calmar"] = evl_data["sharpe"] + evl_data["calmar"]
    evl_data = evl_data.sort_values(by=sort_vars, ascending=False)
    for header_var in header_vars[::-1]:
        evl_data.insert(loc=0, column=header_var, value=evl_data.pop(header_var))

    if sim_type == "facOpt":
        ret_prc = ", ".join(evl_data["ret_prc"].to_list())
        ratio = ", ".join([f"{z:.4f}" for z in evl_data["sharpe+calmar"]])
        print(f"sharpe+calmar:({ret_prc})=({ratio})")

    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", lambda z: f"{z:.4f}")
    print(evl_data)

    check_and_makedirs(evl_save_dir)
    evl_path = os.path.join(evl_save_dir, evl_save_file)
    evl_data.to_csv(evl_path, float_format="%.6f", index=False)
    return 0


"""
------------
--- plot ---
------------
"""


def plot_sim_args_list(
        fig_name: str,
        sim_args_list: list[CSimArgs],
        sim_save_dir: str, plt_save_dir: str,
        bgn_date: str, stp_date: str,
):
    check_and_makedirs(plt_save_dir)
    ret_data_by_sim = {}
    for sim_args in sim_args_list:
        s = CEvlFrmSim(sim_args, sim_save_dir)
        ret_data_by_sim[sim_args.sim_id] = s.get_ret(bgn_date, stp_date)
    ret_data = pd.DataFrame(ret_data_by_sim)
    nav_data = (1 + ret_data).cumprod()
    artist = CPlotLines(
        plot_data=nav_data,
        fig_name=fig_name,
        fig_save_dir=plt_save_dir,
        fig_save_type="jpg",
        colormap="jet",
    )
    artist.plot()
    artist.set_legend()
    artist.set_axis_x(xtick_count=20, xtick_label_size=8)
    artist.add_vlines_from_index(vlines_index=["20240102", "20240902"], color="k")
    artist.save_and_close()
    return 0


def main_plt_grouped_sim_args(
        grouped_sim_args: dict[TSimGrpIdByFacAgg | TRetPrc, list[CSimArgs]],
        sim_save_dir: str,
        plt_save_dir: str,
        bgn_date: str,
        stp_date: str,
):
    for grp_id, sim_args_list in track(grouped_sim_args.items(), description="Plot by group id"):
        if isinstance(grp_id, tuple):
            fig_name = "-".join(grp_id)
        elif isinstance(grp_id, str):
            fig_name = grp_id
        else:
            raise TypeError(f"type of {grp_id} = {type(grp_id)}, is illegal")
        plot_sim_args_list(
            fig_name=fig_name,
            sim_args_list=sim_args_list,
            sim_save_dir=sim_save_dir,
            plt_save_dir=plt_save_dir,
            bgn_date=bgn_date,
            stp_date=stp_date,
        )
    return 0
