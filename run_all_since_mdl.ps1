$bgn_date = "20120104"
$bgn_date_mclrn = "20150601"
$bgn_date_sig = "20150701" # signal bgn date
$bgn_date_sim = "20160104" # simulation bgn date
$stp_date = "20250101"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\mclrn
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\sig_frm_mdl_prd
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\sim_frm_mdl_prd
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\evl_frm_mdl_prd

# ----------------------------
# --- exectue all projects ---
# ----------------------------

# --- machine learning models
python main.py --bgn $bgn_date_mclrn --stp $stp_date mclrn --type parse
python main.py --bgn $bgn_date_mclrn --stp $stp_date --processes 12 mclrn --type trnprd

# --- signals, simulations and evaluations for optimized factors
python main.py --bgn $bgn_date_sig --stp $stp_date signals --type mdlPrd
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type mdlPrd
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type mdlPrd

start E:\Data\Projects\CTA_ML_Tushare_V3\evl_frm_mdl_prd\plot-nav\Cls.jpg
start E:\Data\Projects\CTA_ML_Tushare_V3\evl_frm_mdl_prd\plot-nav\Opn.jpg
