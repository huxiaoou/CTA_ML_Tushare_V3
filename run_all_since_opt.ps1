$bgn_date = "20120104"
$bgn_date_mclrn = "20150601"
$bgn_date_sig = "20150701" # signal bgn date
$bgn_date_sim = "20160104" # simulation bgn date
$stp_date = "20241101"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\opt_frm_slc_fac
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\sig_frm_fac_opt
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\sim_frm_fac_opt
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\evl_frm_fac_opt

# ----------------------------
# --- exectue all projects ---
# ----------------------------

# --- optimize for selected factors
python main.py --bgn $bgn_date_sig --stp $stp_date optimize --type slcFac

# --- signals, simulations and evaluations for optimized factors
python main.py --bgn $bgn_date_sig --stp $stp_date signals --type facOpt
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type facOpt
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type facOpt

start E:\Data\Projects\CTA_ML_Tushare_V3\evl_frm_fac_opt\plot-nav\Cls.Opn.Omega.jpg
