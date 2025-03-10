import os
import yaml
from husfort.qsqlite import CDbStruct, CSqlTable
from typedef import TUniverse, TInstruName, CCfgInstru, CCfgAvlbUnvrs, CCfgConst, CCfgTrn, CCfgPrd, CCfgSim, CCfgDecay
from typedef import CCfgProj, CCfgDbStruct
from typedef import (
    CCfgFactors,
    CCfgFactorMTM,
    CCfgFactorSKEW,
    CCfgFactorKURT,
    CCfgFactorRS,
    CCfgFactorBASIS,
    CCfgFactorTS,
    CCfgFactorS0BETA,
    CCfgFactorS1BETA,
    CCfgFactorCBETA,
    CCfgFactorIBETA,
    CCfgFactorPBETA,
    CCfgFactorCTP,
    CCfgFactorCTR,
    CCfgFactorCVP,
    CCfgFactorCVR,
    CCfgFactorCSP,
    CCfgFactorCSR,
    CCfgFactorCOV,
    CCfgFactorNOI,
    CCfgFactorNDOI,
    CCfgFactorWNOI,
    CCfgFactorWNDOI,
    CCfgFactorSPDWEB,
    CCfgFactorSIZE,
    CCfgFactorHR,
    CCfgFactorSR,
    CCfgFactorLIQUIDITY,
    CCfgFactorVSTD,
    CCfgFactorAMP,
    CCfgFactorEXR,
    CCfgFactorSMT,
    CCfgFactorRWTC,
    CCfgFactorTAILS,
    CCfgFactorHEADS,
    CCfgFactorTOPS,
    CCfgFactorDOV,
    CCfgFactorRES,
    CCfgFactorVOL,
    CCfgFactorMF,
    CCfgFactorRV,
    CCfgFactorTA,
)

# ---------- project configuration ----------

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

universe = TUniverse({TInstruName(k): CCfgInstru(**v) for k, v in _config["universe"].items()})

proj_cfg = CCfgProj(
    # --- shared
    calendar_path=_config["path"]["calendar_path"],
    root_dir=_config["path"]["root_dir"],
    db_struct_path=_config["path"]["db_struct_path"],
    alternative_dir=_config["path"]["alternative_dir"],
    market_index_path=_config["path"]["market_index_path"],
    by_instru_pos_dir=_config["path"]["by_instru_pos_dir"],
    by_instru_pre_dir=_config["path"]["by_instru_pre_dir"],
    by_instru_min_dir=_config["path"]["by_instru_min_dir"],

    # --- project
    project_root_dir=_config["path"]["project_root_dir"],
    available_dir=os.path.join(_config["path"]["project_root_dir"], _config["path"]["available_dir"]),  # type:ignore
    market_dir=os.path.join(_config["path"]["project_root_dir"], _config["path"]["market_dir"]),  # type:ignore
    test_return_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["test_return_dir"]),
    factors_by_instru_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["factors_by_instru_dir"]),
    factors_aggr_avlb_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["factors_aggr_avlb_dir"]),

    sig_frm_fac_agg_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sig_frm_fac_agg_dir"]),
    sim_frm_fac_agg_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sim_frm_fac_agg_dir"]),
    evl_frm_fac_agg_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["evl_frm_fac_agg_dir"]),
    opt_frm_slc_fac_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["opt_frm_slc_fac_dir"]),

    sig_frm_fac_opt_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sig_frm_fac_opt_dir"]),
    sim_frm_fac_opt_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sim_frm_fac_opt_dir"]),
    evl_frm_fac_opt_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["evl_frm_fac_opt_dir"]),

    sig_frm_mdl_prd_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sig_frm_mdl_prd_dir"]),
    sim_frm_mdl_prd_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["sim_frm_mdl_prd_dir"]),
    evl_frm_mdl_prd_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["evl_frm_mdl_prd_dir"]),

    mclrn_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["mclrn_dir"],
    ),
    mclrn_cfg_file=_config["path"]["mclrn_cfg_file"],
    mclrn_mdl_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["mclrn_dir"], _config["path"]["mclrn_mdl_dir"],
    ),
    mclrn_prd_dir=os.path.join(  # type:ignore
        _config["path"]["project_root_dir"], _config["path"]["mclrn_dir"], _config["path"]["mclrn_prd_dir"],
    ),

    universe=universe,
    avlb_unvrs=CCfgAvlbUnvrs(**_config["available"]),
    mkt_idxes=_config["mkt_idxes"],
    const=CCfgConst(**_config["CONST"]),

    trn=CCfgTrn(**_config["trn"]),
    prd=CCfgPrd(**_config["prd"]),
    sim=CCfgSim(**_config["sim"]),
    decay=CCfgDecay(**_config["decay"]),
    optimize=_config["optimize"],
    factors=_config["factors"],
    selected_factors_pool=_config["selected_factors_pool"],
    cv=_config["cv"],
    mclrn=_config["mclrn"],
    omega=_config["omega"],
)

# ---------- databases structure ----------
with open(proj_cfg.db_struct_path, "r") as f:
    _db_struct = yaml.safe_load(f)

db_struct_cfg = CCfgDbStruct(
    # --- shared database
    macro=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
        db_name=_db_struct["macro"]["db_name"],
        table=CSqlTable(cfg=_db_struct["macro"]["table"]),
    ),
    forex=CDbStruct(
        db_save_dir=proj_cfg.alternative_dir,
        db_name=_db_struct["forex"]["db_name"],
        table=CSqlTable(cfg=_db_struct["forex"]["table"]),
    ),
    fmd=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["fmd"]["db_name"],
        table=CSqlTable(cfg=_db_struct["fmd"]["table"]),
    ),
    position=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["position"]["db_name"],
        table=CSqlTable(cfg=_db_struct["position"]["table"]),
    ),
    basis=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["basis"]["db_name"],
        table=CSqlTable(cfg=_db_struct["basis"]["table"]),
    ),
    stock=CDbStruct(
        db_save_dir=proj_cfg.root_dir,
        db_name=_db_struct["stock"]["db_name"],
        table=CSqlTable(cfg=_db_struct["stock"]["table"]),
    ),
    preprocess=CDbStruct(
        db_save_dir=proj_cfg.by_instru_pre_dir,
        db_name=_db_struct["preprocess"]["db_name"],
        table=CSqlTable(cfg=_db_struct["preprocess"]["table"]),
    ),
    minute_bar=CDbStruct(
        db_save_dir=proj_cfg.by_instru_min_dir,
        db_name=_db_struct["fMinuteBar"]["db_name"],
        table=CSqlTable(cfg=_db_struct["fMinuteBar"]["table"]),
    ),

    # --- project database
    available=CDbStruct(
        db_save_dir=proj_cfg.available_dir,
        db_name=_config["db_struct"]["available"]["db_name"],
        table=CSqlTable(cfg=_config["db_struct"]["available"]["table"]),
    ),
    market=CDbStruct(
        db_save_dir=proj_cfg.market_dir,
        db_name=_config["db_struct"]["market"]["db_name"],
        table=CSqlTable(cfg=_config["db_struct"]["market"]["table"]),
    ),
)

# --- factors ---
cfg_factors = CCfgFactors(
    MTM=CCfgFactorMTM(**proj_cfg.factors["MTM"]),
    SKEW=CCfgFactorSKEW(**proj_cfg.factors["SKEW"]),
    KURT=CCfgFactorKURT(**proj_cfg.factors["KURT"]),
    RS=CCfgFactorRS(**proj_cfg.factors["RS"]),
    BASIS=CCfgFactorBASIS(**proj_cfg.factors["BASIS"]),
    TS=CCfgFactorTS(**proj_cfg.factors["TS"]),
    S0BETA=CCfgFactorS0BETA(**proj_cfg.factors["S0BETA"]),
    S1BETA=CCfgFactorS1BETA(**proj_cfg.factors["S1BETA"]),
    CBETA=CCfgFactorCBETA(**proj_cfg.factors["CBETA"]),
    IBETA=CCfgFactorIBETA(**proj_cfg.factors["IBETA"]),
    PBETA=CCfgFactorPBETA(**proj_cfg.factors["PBETA"]),
    CTP=CCfgFactorCTP(**proj_cfg.factors["CTP"]),
    CTR=None,  # CCfgFactorCTR(**proj_cfg.factors["CTR"]),
    CVP=CCfgFactorCVP(**proj_cfg.factors["CVP"]),
    CVR=None,  # CCfgFactorCVR(**proj_cfg.factors["CVR"]),
    CSP=CCfgFactorCSP(**proj_cfg.factors["CSP"]),
    CSR=None,  # CCfgFactorCSR(**proj_cfg.factors["CSR"]),
    COV=CCfgFactorCOV(**proj_cfg.factors["COV"]),
    NOI=CCfgFactorNOI(**proj_cfg.factors["NOI"]),
    NDOI=CCfgFactorNDOI(**proj_cfg.factors["NDOI"]),
    WNOI=None,  # CCfgFactorWNOI(**proj_cfg.factors["WNOI"]),
    WNDOI=None,  # CCfgFactorWNDOI(**proj_cfg.factors["WNDOI"]),
    SPDWEB=CCfgFactorSPDWEB(**proj_cfg.factors["SPDWEB"]),
    SIZE=CCfgFactorSIZE(**proj_cfg.factors["SIZE"]),
    HR=CCfgFactorHR(**proj_cfg.factors["HR"]),
    SR=CCfgFactorSR(**proj_cfg.factors["SR"]),
    LIQUIDITY=CCfgFactorLIQUIDITY(**proj_cfg.factors["LIQUIDITY"]),
    VSTD=CCfgFactorVSTD(**proj_cfg.factors["VSTD"]),
    AMP=CCfgFactorAMP(**proj_cfg.factors["AMP"]),
    EXR=CCfgFactorEXR(**proj_cfg.factors["EXR"]),
    SMT=CCfgFactorSMT(**proj_cfg.factors["SMT"]),
    RWTC=CCfgFactorRWTC(**proj_cfg.factors["RWTC"]),
    TAILS=CCfgFactorTAILS(**proj_cfg.factors["TAILS"]),
    HEADS=CCfgFactorHEADS(**proj_cfg.factors["HEADS"]),
    TOPS=CCfgFactorTOPS(**proj_cfg.factors["TOPS"]),
    TA=CCfgFactorTA(**proj_cfg.factors["TA"]),
    DOV=CCfgFactorDOV(**proj_cfg.factors["DOV"]),
    RES=CCfgFactorRES(**proj_cfg.factors["RES"]),
    VOL=CCfgFactorVOL(**proj_cfg.factors["VOL"]),
    MF=CCfgFactorMF(**proj_cfg.factors["MF"]),
    RV=CCfgFactorRV(**proj_cfg.factors["RV"]),
)

if __name__ == "__main__":
    sep = "-" * 80

    print(sep)
    print(f"Size of universe = {len(universe)}")
    for instru, sectors in universe.items():
        print(f"{instru:>6s}: {sectors}")

    print(sep)
    d = {k: v for k, v in vars(cfg_factors).items() if v is not None}
    print(f"Size of activated factors class = {len(d)}")
    for factor, cfg in d.items():
        print(f"{factor:>6s}: {cfg}")

    print(sep)
    factors = cfg_factors.get_factors()
    print(f"Size of raw factors = {len(factors)}")
    for factor in factors:
        print(factor)

    print(sep)
    selected_factors = proj_cfg.selected_factors_pool
    print(f"Size of selected factors = {len(selected_factors)}")
    for factor in selected_factors:
        print(factor)
