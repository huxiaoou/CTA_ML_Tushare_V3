import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="To calculate data, such as macro and forex")
    arg_parser.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    arg_parser.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")
    arg_parser.add_argument("--nomp", default=False, action="store_true",
                            help="not using multiprocess, for debug. Works only when switch in (factor,)")
    arg_parser.add_argument("--processes", type=int, default=None,
                            help="number of processes to be called, effective only when nomp = False")
    arg_parser.add_argument("--verbose", default=False, action="store_true",
                            help="whether to print more details, effective only when sub function = (feature_selection,)")

    arg_parser_subs = arg_parser.add_subparsers(
        title="Position argument to call sub functions",
        dest="switch",
        description="use this position argument to call different functions of this project. "
                    "For example: 'python main.py --bgn 20120104 --stp 20240826 available'",
        required=True,
    )

    # switch: available
    arg_parser_subs.add_parser(name="available", help="Calculate available universe")

    # switch: market
    arg_parser_subs.add_parser(name="market", help="Calculate market universe")

    # switch: test return
    arg_parser_subs.add_parser(name="test_return", help="Calculate test returns")

    # switch: factor
    arg_parser_sub = arg_parser_subs.add_parser(name="factor", help="Calculate factor")
    arg_parser_sub.add_argument(
        "--fclass", type=str, help="factor class to run", required=True,
        choices=("MTM", "SKEW", "KURT",
                 "RS", "BASIS", "TS",
                 "S0BETA", "S1BETA", "CBETA", "IBETA", "PBETA",
                 "CTP", "CTR", "CVP", "CVR", "CSP", "CSR", "COV",
                 "NOI", "NDOI", "WNOI", "WNDOI", "SPDWEB",
                 "SIZE", "HR", "SR", "LIQUIDITY", "VSTD",
                 "AMP", "EXR", "SMT", "RWTC", "TAILS", "HEADS",
                 "TOPS", "DOV", "RES", "VOL", "MF", "RV",
                 "TA",),
    )

    # switch: signals
    arg_parser_sub = arg_parser_subs.add_parser(name="signals", help="generate signals")
    arg_parser_sub.add_argument("--type", type=str, choices=("facAgg", "facOpt", "mdlPrd"))

    # switch: simulations
    arg_parser_sub = arg_parser_subs.add_parser(name="simulations", help="simulate from signals")
    arg_parser_sub.add_argument("--type", type=str, choices=("facAgg", "facOpt", "mdlPrd"))

    # switch: evaluations
    arg_parser_sub = arg_parser_subs.add_parser(name="evaluations", help="evaluate simulations")
    arg_parser_sub.add_argument("--type", type=str, choices=("facAgg", "facOpt", "mdlPrd"))

    # switch: optimize
    arg_parser_sub = arg_parser_subs.add_parser(name="optimize", help="optimize portfolio and signals")
    arg_parser_sub.add_argument("--type", type=str, choices=("slcFac",))

    # switch: mclrn
    arg_parser_sub = arg_parser_subs.add_parser(name="mclrn", help="machine learning functions")
    arg_parser_sub.add_argument("--type", type=str, choices=("parse", "trnprd"))

    return arg_parser.parse_args()


if __name__ == "__main__":
    import os
    from project_config import proj_cfg, db_struct_cfg, cfg_factors
    from husfort.qlog import define_logger
    from husfort.qcalendar import CCalendar

    define_logger()

    calendar = CCalendar(proj_cfg.calendar_path)
    args = parse_args()
    bgn_date, stp_date = args.bgn, args.stp or calendar.get_next_date(args.bgn, shift=1)

    if args.switch == "available":
        from solutions.available import main_available

        main_available(
            bgn_date=bgn_date, stp_date=stp_date,
            universe=proj_cfg.universe,
            cfg_avlb_unvrs=proj_cfg.avlb_unvrs,
            db_struct_preprocess=db_struct_cfg.preprocess,
            db_struct_avlb=db_struct_cfg.available,
            calendar=calendar,
        )
    elif args.switch == "market":
        from solutions.market import main_market

        main_market(
            bgn_date=bgn_date, stp_date=stp_date,
            calendar=calendar,
            db_struct_avlb=db_struct_cfg.available,
            db_struct_mkt=db_struct_cfg.market,
            path_mkt_idx_data=proj_cfg.market_index_path,
            mkt_idxes=list(proj_cfg.mkt_idxes.values()),
            sectors=proj_cfg.const.SECTORS,
        )
    elif args.switch == "test_return":
        from solutions.test_return import CTstRetRaw, CTstRetAgg

        for win in proj_cfg.test_rets_wins:
            # --- raw return
            tst_ret_raw = CTstRetRaw(
                win=win, lag=proj_cfg.const.LAG,
                universe=proj_cfg.universe,
                db_tst_ret_save_dir=proj_cfg.test_return_dir,
                db_struct_preprocess=db_struct_cfg.preprocess,
            )
            tst_ret_raw.main_test_return_raw(bgn_date, stp_date, calendar)

            # --- aggregate
            tst_ret_agg = CTstRetAgg(
                win=win, lag=proj_cfg.const.LAG,
                universe=proj_cfg.universe,
                db_tst_ret_save_dir=proj_cfg.test_return_dir,
                db_struct_avlb=db_struct_cfg.available,
            )
            tst_ret_agg.main_test_return_agg(
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
            )
    elif args.switch == "factor":
        from project_config import cfg_factors

        fac, fclass = None, args.fclass
        if fclass == "MTM":
            if (cfg := cfg_factors.MTM) is not None:
                from solutions.factorAlg import CFactorMTM

                fac = CFactorMTM(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "SKEW":
            if (cfg := cfg_factors.SKEW) is not None:
                from solutions.factorAlg import CFactorSKEW

                fac = CFactorSKEW(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "KURT":
            if (cfg := cfg_factors.KURT) is not None:
                from solutions.factorAlg import CFactorKURT

                fac = CFactorKURT(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "RS":
            if (cfg := cfg_factors.RS) is not None:
                from solutions.factorAlg import CFactorRS

                fac = CFactorRS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "BASIS":
            if (cfg := cfg_factors.BASIS) is not None:
                from solutions.factorAlg import CFactorBASIS

                fac = CFactorBASIS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "TS":
            if (cfg := cfg_factors.TS) is not None:
                from solutions.factorAlg import CFactorTS

                fac = CFactorTS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "S0BETA":
            if (cfg := cfg_factors.S0BETA) is not None:
                from solutions.factorAlg import CFactorS0BETA

                fac = CFactorS0BETA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_mkt=db_struct_cfg.market,
                )
        elif fclass == "S1BETA":
            if (cfg := cfg_factors.S1BETA) is not None:
                from solutions.factorAlg import CFactorS1BETA

                fac = CFactorS1BETA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_mkt=db_struct_cfg.market,
                )
        elif fclass == "CBETA":
            if (cfg := cfg_factors.CBETA) is not None:
                from solutions.factorAlg import CFactorCBETA

                fac = CFactorCBETA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_forex=db_struct_cfg.forex,
                )
        elif fclass == "IBETA":
            if (cfg := cfg_factors.IBETA) is not None:
                from solutions.factorAlg import CFactorIBETA

                fac = CFactorIBETA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_macro=db_struct_cfg.macro,
                )
        elif fclass == "PBETA":
            if (cfg := cfg_factors.PBETA) is not None:
                from solutions.factorAlg import CFactorPBETA

                fac = CFactorPBETA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_macro=db_struct_cfg.macro,
                )
        elif fclass == "CTP":
            if (cfg := cfg_factors.CTP) is not None:
                from solutions.factorAlg import CFactorCTP

                fac = CFactorCTP(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "CTR":
            if (cfg := cfg_factors.CTR) is not None:
                from solutions.factorAlg import CFactorCTR

                fac = CFactorCTR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "CVP":
            if (cfg := cfg_factors.CVP) is not None:
                from solutions.factorAlg import CFactorCVP

                fac = CFactorCVP(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "CVR":
            if (cfg := cfg_factors.CVR) is not None:
                from solutions.factorAlg import CFactorCVR

                fac = CFactorCVR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "CSP":
            if (cfg := cfg_factors.CSP) is not None:
                from solutions.factorAlg import CFactorCSP

                fac = CFactorCSP(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "CSR":
            if (cfg := cfg_factors.CSR) is not None:
                from solutions.factorAlg import CFactorCSR

                fac = CFactorCSR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "COV":
            if (cfg := cfg_factors.COV) is not None:
                from solutions.factorAlg import CFactorCOV

                fac = CFactorCOV(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "NOI":
            if (cfg := cfg_factors.NOI) is not None:
                from solutions.factorAlg import CFactorNOI

                fac = CFactorNOI(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_pos=db_struct_cfg.position.copy_to_another(
                        another_db_save_dir=proj_cfg.by_instru_pos_dir),
                )
        elif fclass == "NDOI":
            if (cfg := cfg_factors.NDOI) is not None:
                from solutions.factorAlg import CFactorNDOI

                fac = CFactorNDOI(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_pos=db_struct_cfg.position.copy_to_another(
                        another_db_save_dir=proj_cfg.by_instru_pos_dir),
                )
        elif fclass == "WNOI":
            if (cfg := cfg_factors.WNOI) is not None:
                from solutions.factorAlg import CFactorWNOI

                fac = CFactorWNOI(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_pos=db_struct_cfg.position.copy_to_another(
                        another_db_save_dir=proj_cfg.by_instru_pos_dir),
                )
        elif fclass == "WNDOI":
            if (cfg := cfg_factors.WNDOI) is not None:
                from solutions.factorAlg import CFactorWNDOI

                fac = CFactorWNDOI(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_pos=db_struct_cfg.position.copy_to_another(
                        another_db_save_dir=proj_cfg.by_instru_pos_dir),
                )
        elif fclass == "SPDWEB":
            if (cfg := cfg_factors.SPDWEB) is not None:
                from solutions.factorAlg import CFactorSPDWEB

                fac = CFactorSPDWEB(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_pos=db_struct_cfg.position.copy_to_another(
                        another_db_save_dir=proj_cfg.by_instru_pos_dir),
                )
        elif fclass == "SIZE":
            if (cfg := cfg_factors.SIZE) is not None:
                from solutions.factorAlg import CFactorSIZE

                fac = CFactorSIZE(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "HR":
            if (cfg := cfg_factors.HR) is not None:
                from solutions.factorAlg import CFactorHR

                fac = CFactorHR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "SR":
            if (cfg := cfg_factors.SR) is not None:
                from solutions.factorAlg import CFactorSR

                fac = CFactorSR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "LIQUIDITY":
            if (cfg := cfg_factors.LIQUIDITY) is not None:
                from solutions.factorAlg import CFactorLIQUIDITY

                fac = CFactorLIQUIDITY(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "VSTD":
            if (cfg := cfg_factors.VSTD) is not None:
                from solutions.factorAlg import CFactorVSTD

                fac = CFactorVSTD(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "AMP":
            if (cfg := cfg_factors.AMP) is not None:
                from solutions.factorAlg import CFactorAMP

                fac = CFactorAMP(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "EXR":
            if (cfg := cfg_factors.EXR) is not None:
                from solutions.factorAlg import CFactorEXR

                fac = CFactorEXR(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "SMT":
            if (cfg := cfg_factors.SMT) is not None:
                from solutions.factorAlg import CFactorSMT

                fac = CFactorSMT(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "RWTC":
            if (cfg := cfg_factors.RWTC) is not None:
                from solutions.factorAlg import CFactorRWTC

                fac = CFactorRWTC(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "TAILS":
            if (cfg := cfg_factors.TAILS) is not None:
                from solutions.factorAlg import CFactorTAILS

                fac = CFactorTAILS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "HEADS":
            if (cfg := cfg_factors.HEADS) is not None:
                from solutions.factorAlg import CFactorHEADS

                fac = CFactorHEADS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "TOPS":
            if (cfg := cfg_factors.TOPS) is not None:
                from solutions.factorAlg import CFactorTOPS

                fac = CFactorTOPS(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "DOV":
            if (cfg := cfg_factors.DOV) is not None:
                from solutions.factorAlg import CFactorDOV

                fac = CFactorDOV(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "RES":
            if (cfg := cfg_factors.RES) is not None:
                from solutions.factorAlg import CFactorRES

                fac = CFactorRES(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "VOL":
            if (cfg := cfg_factors.VOL) is not None:
                from solutions.factorAlg import CFactorVOL

                fac = CFactorVOL(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                )
        elif fclass == "MF":
            if (cfg := cfg_factors.MF) is not None:
                from solutions.factorAlg import CFactorMF

                fac = CFactorMF(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "RV":
            if (cfg := cfg_factors.RV) is not None:
                from solutions.factorAlg import CFactorRV

                fac = CFactorRV(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        elif fclass == "TA":
            if (cfg := cfg_factors.TA) is not None:
                from solutions.factorAlg import CFactorTA

                fac = CFactorTA(
                    cfg=cfg,
                    factors_by_instru_dir=proj_cfg.factors_by_instru_dir,
                    universe=proj_cfg.universe,
                    db_struct_preprocess=db_struct_cfg.preprocess,
                    db_struct_minute_bar=db_struct_cfg.minute_bar,
                )
        else:
            raise NotImplementedError(f"fclass = {args.fclass}")

        if fac is not None:
            from solutions.factor import CFactorAgg

            # --- raw factors
            fac.main_raw(
                bgn_date=bgn_date, stp_date=stp_date, calendar=calendar,
                call_multiprocess=not args.nomp, processes=args.processes,
            )

            # --- aggregation
            aggregator = CFactorAgg(
                ref_factor=fac,
                universe=proj_cfg.universe,
                db_struct_preprocess=db_struct_cfg.preprocess,
                db_struct_avlb=db_struct_cfg.available,
                factors_aggr_avlb_dir=proj_cfg.factors_aggr_avlb_dir,
            )
            aggregator.main_agg(bgn_date=bgn_date, stp_date=stp_date, calendar=calendar)
    elif args.switch == "signals":
        if args.type == "facAgg":
            from solutions.signals import main_signals_from_factor_agg

            factors = cfg_factors.get_factors()
            main_signals_from_factor_agg(
                factors=factors,
                factor_save_root_dir=proj_cfg.factors_aggr_avlb_dir,
                decay_rate=proj_cfg.decay.rate,
                decay_win=proj_cfg.decay.win,
                signal_save_dir=proj_cfg.sig_frm_fac_agg_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
        elif args.type == "facOpt":
            from solutions.shared import group_sim_args_from_slc_fac
            from solutions.signals import main_signals_from_opt

            grouped_sim_args = group_sim_args_from_slc_fac(
                factor_names=proj_cfg.selected_factors_pool,
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_agg_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST_SUB,
            )
            main_signals_from_opt(
                grouped_sim_args=grouped_sim_args,
                input_sig_dir=proj_cfg.sig_frm_fac_agg_dir,
                input_opt_dir=proj_cfg.opt_frm_slc_fac_dir,
                signal_save_dir=proj_cfg.sig_frm_fac_opt_dir,
                db_struct_avlb=db_struct_cfg.available,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
        elif args.type == "mdlPrd":
            from solutions.mclrn_mdl_parser import load_config_models
            from solutions.shared import gen_model_tests
            from solutions.signals import main_signals_from_mdl_prd

            config_models = load_config_models(cfg_mdl_dir=proj_cfg.mclrn_dir, cfg_mdl_file=proj_cfg.mclrn_cfg_file)
            factors = cfg_factors.get_factor_from_names(proj_cfg.selected_factors_pool)
            test_mdls = gen_model_tests(config_models=config_models, factors=factors)

            main_signals_from_mdl_prd(
                tests=test_mdls,
                mclrn_prd_dir=proj_cfg.mclrn_prd_dir,
                signal_save_dir=proj_cfg.sig_frm_mdl_prd_dir,
                decay_win=proj_cfg.decay.win,
                decay_rate=proj_cfg.decay.rate,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
        else:
            raise ValueError(f"args.type == {args.type} is illegal")
    elif args.switch == "simulations":
        from solutions.simulations import main_simulations

        if args.type == "facAgg":
            from solutions.shared import get_sim_args_fac

            sim_args_list = get_sim_args_fac(
                factors=cfg_factors.get_factors(),
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_agg_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST_SUB,
            )
            main_simulations(
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_agg_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
        elif args.type == "facOpt":
            from solutions.shared import get_sim_args_fac_opt
            from solutions.simulations import main_simu_from_nav

            sim_args_list = get_sim_args_fac_opt(
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_opt_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST,
            )
            main_simulations(
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_opt_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
            main_simu_from_nav(
                save_id=proj_cfg.omega["pid"],
                portfolio_ids=proj_cfg.omega["components"],
                bgn_date=bgn_date,
                stp_date=stp_date,
                sim_save_dir=proj_cfg.sim_frm_fac_opt_dir,
                calendar=calendar,
            )
        elif args.type == "mdlPrd":
            from solutions.mclrn_mdl_parser import load_config_models
            from solutions.shared import gen_model_tests, get_sim_args_mdl_prd

            config_models = load_config_models(cfg_mdl_dir=proj_cfg.mclrn_dir, cfg_mdl_file=proj_cfg.mclrn_cfg_file)
            factors = cfg_factors.get_factor_from_names(proj_cfg.selected_factors_pool)
            test_mdls = gen_model_tests(config_models=config_models, factors=factors)
            sim_args_list = get_sim_args_mdl_prd(
                tests=test_mdls,
                signals_dir=proj_cfg.sig_frm_mdl_prd_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST,
            )
            main_simulations(
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_mdl_prd_dir,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
        else:
            raise ValueError(f"args.type == {args.type} is illegal")
    elif args.switch == "evaluations":
        from solutions.evaluations import main_evl_sims, main_plt_grouped_sim_args, plot_sim_args_list

        if args.type == "facAgg":
            from solutions.shared import get_sim_args_fac, group_sim_args_by_factor_class

            sim_args_list = get_sim_args_fac(
                factors=cfg_factors.get_factors(),
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_agg_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST_SUB,
            )
            main_evl_sims(
                sim_type=args.type,
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_agg_dir,
                evl_save_dir=proj_cfg.evl_frm_fac_agg_dir,
                evl_save_file="evaluations_for_fac_agg.csv.gz",
                header_vars=["sharpe", "calmar", "sharpe+calmar"],
                sort_vars=["sharpe"],
                bgn_date=bgn_date,
                stp_date=stp_date,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
            # plot by group
            grouped_sim_args = group_sim_args_by_factor_class(sim_args_list, cfg_factors.get_mapper_name_to_class())
            main_plt_grouped_sim_args(
                grouped_sim_args=grouped_sim_args,
                sim_save_dir=proj_cfg.sim_frm_fac_agg_dir,
                plt_save_dir=os.path.join(proj_cfg.evl_frm_fac_agg_dir, "plot-nav"),
                bgn_date=bgn_date,
                stp_date=stp_date,
            )
        elif args.type == "facOpt":
            from solutions.shared import get_sim_args_fac_opt, get_sim_args_omega

            sim_args_list = get_sim_args_fac_opt(
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_opt_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST,
            ) + [get_sim_args_omega(omega_id=proj_cfg.omega["pid"])]
            main_evl_sims(
                sim_type=args.type,
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_opt_dir,
                evl_save_dir=proj_cfg.evl_frm_fac_opt_dir,
                evl_save_file="evaluations_for_fac_opt.csv.gz",
                header_vars=["sharpe+calmar", "sharpe", "calmar"],
                sort_vars=["sharpe+calmar"],
                bgn_date=bgn_date,
                stp_date=stp_date,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
            plot_sim_args_list(
                fig_name="Cls.Opn.Omega",
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_opt_dir,
                plt_save_dir=os.path.join(proj_cfg.evl_frm_fac_opt_dir, "plot-nav"),
                bgn_date=bgn_date, stp_date=stp_date,
                line_style=["-.", "-.", "-"],
                line_color=["#000080", "#006400", "#8B0000"],
                vlines_index=["20231229", "20240902", "20241231"],
            )
            plot_sim_args_list(
                fig_name="Cls.Opn.Omega.since_2025",
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_fac_opt_dir,
                plt_save_dir=os.path.join(proj_cfg.evl_frm_fac_opt_dir, "plot-nav"),
                bgn_date="20250102", stp_date=stp_date,
                line_style=["-.", "-.", "-"],
                line_color=["#000080", "#006400", "#8B0000"],
            )
        elif args.type == "mdlPrd":
            from solutions.mclrn_mdl_parser import load_config_models
            from solutions.shared import gen_model_tests, get_sim_args_mdl_prd, group_sim_args_by_ret_prc

            config_models = load_config_models(cfg_mdl_dir=proj_cfg.mclrn_dir, cfg_mdl_file=proj_cfg.mclrn_cfg_file)
            factors = cfg_factors.get_factor_from_names(proj_cfg.selected_factors_pool)
            test_mdls = gen_model_tests(config_models=config_models, factors=factors)
            sim_args_list = get_sim_args_mdl_prd(
                tests=test_mdls,
                signals_dir=proj_cfg.sig_frm_mdl_prd_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST_SUB,
            )
            main_evl_sims(
                sim_type=args.type,
                sim_args_list=sim_args_list,
                sim_save_dir=proj_cfg.sim_frm_mdl_prd_dir,
                evl_save_dir=proj_cfg.evl_frm_mdl_prd_dir,
                evl_save_file="evaluations_for_mdl_prd.csv.gz",
                header_vars=["sharpe+calmar", "sharpe", "calmar"],
                sort_vars=["sharpe+calmar"],
                bgn_date=bgn_date,
                stp_date=stp_date,
                call_multiprocess=not args.nomp,
                processes=args.processes,
            )
            # plot by group
            grouped_sim_args = group_sim_args_by_ret_prc(sim_args_list)
            main_plt_grouped_sim_args(
                grouped_sim_args=grouped_sim_args,
                sim_save_dir=proj_cfg.sim_frm_mdl_prd_dir,
                plt_save_dir=os.path.join(proj_cfg.evl_frm_mdl_prd_dir, "plot-nav"),
                bgn_date=bgn_date,
                stp_date=stp_date,
            )
        else:
            raise ValueError(f"args.type == {args.type} is illegal")
    elif args.switch == "optimize":
        from solutions.optimize import main_optimize

        if args.type == "slcFac":
            from solutions.shared import group_sim_args_from_slc_fac

            grouped_sim_args = group_sim_args_from_slc_fac(
                factor_names=proj_cfg.selected_factors_pool,
                rets=proj_cfg.get_test_rets(),
                signals_dir=proj_cfg.sig_frm_fac_agg_dir,
                ret_dir=proj_cfg.test_return_dir,
                cost=proj_cfg.const.COST_SUB,
            )
            main_optimize(
                grouped_sim_args=grouped_sim_args,
                sim_save_dir=proj_cfg.sim_frm_fac_agg_dir,
                lbd=proj_cfg.optimize["lbd"],
                win=proj_cfg.optimize["win"],
                save_dir=proj_cfg.opt_frm_slc_fac_dir,
                bgn_date=bgn_date, stp_date=stp_date, calendar=calendar,
            )
        else:
            raise ValueError(f"args.type == {args.type} is illegal")
    elif args.switch == "mclrn":
        if args.type == "parse":
            from solutions.mclrn_mdl_parser import parse_model_configs

            parse_model_configs(
                models=proj_cfg.mclrn,
                rets=proj_cfg.get_test_rets(),
                trn_wins=proj_cfg.trn.wins,
                cfg_mdl_dir=proj_cfg.mclrn_dir,
                cfg_mdl_file=proj_cfg.mclrn_cfg_file,
            )
        elif args.type == "trnprd":
            from solutions.mclrn_mdl_parser import load_config_models
            from solutions.shared import gen_model_tests
            from solutions.mclrn_mdl_trn_prd import main_train_and_predict

            config_models = load_config_models(cfg_mdl_dir=proj_cfg.mclrn_dir, cfg_mdl_file=proj_cfg.mclrn_cfg_file)
            factors = cfg_factors.get_factor_from_names(proj_cfg.selected_factors_pool)
            test_mdls = gen_model_tests(config_models=config_models, factors=factors)
            main_train_and_predict(
                tests=test_mdls,
                cv=proj_cfg.cv,
                factors_save_root_dir=proj_cfg.factors_aggr_avlb_dir,
                tst_ret_save_root_dir=proj_cfg.test_return_dir,
                db_struct_avlb=db_struct_cfg.available,
                mclrn_mdl_dir=proj_cfg.mclrn_mdl_dir,
                mclrn_prd_dir=proj_cfg.mclrn_prd_dir,
                universe=proj_cfg.universe,
                bgn_date=bgn_date,
                stp_date=stp_date,
                calendar=calendar,
                call_multiprocess=not args.nomp,
                processes=args.processes,
                verbose=args.verbose,
            )
        else:
            raise ValueError(f"args.type == {args.type} is illegal")
    else:
        raise ValueError(f"args.switch = {args.switch} is illegal")
