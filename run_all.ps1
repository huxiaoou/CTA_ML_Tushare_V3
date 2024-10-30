$bgn_date = "20120104"
$bgn_date_sig = "20170703" # signal bgn date
$bgn_date_sim = "20180102" # simulation bgn date
$stp_date = "20241008"

$bgn_date_ml = "20170201" # machine learning bgn date
$bgn_date_mdl_prd = "20170301"
$bgn_date_mdl_opt = "20170405"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare_V3\*

# ----------------------------
# --- exectue all projects ---
# ----------------------------

# --- prepare
python main.py --bgn $bgn_date --stp $stp_date available
python main.py --bgn $bgn_date --stp $stp_date market
python main.py --bgn $bgn_date --stp $stp_date test_return

# --- factor
python main.py --bgn $bgn_date --stp $stp_date factor --fclass MTM
python main.py --bgn $bgn_date --stp $stp_date factor --fclass SKEW
python main.py --bgn $bgn_date --stp $stp_date factor --fclass RS
python main.py --bgn $bgn_date --stp $stp_date factor --fclass BASIS
python main.py --bgn $bgn_date --stp $stp_date factor --fclass TS
python main.py --bgn $bgn_date --stp $stp_date factor --fclass S0BETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass S1BETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass IBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass PBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CTP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CTR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CVP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CVR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CSP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CSR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass NOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass NDOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass WNOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass WNDOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass AMP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass EXR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass SMT
python main.py --bgn $bgn_date --stp $stp_date factor --fclass RWTC
python main.py --bgn $bgn_date --stp $stp_date factor --fclass TA

# --- single factor test
python main.py --bgn $bgn_date_sig --stp $stp_date signals --type facNeu
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type facNeu
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type facNeu
