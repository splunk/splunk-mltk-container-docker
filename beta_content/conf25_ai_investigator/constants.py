#---------------------------------------------------------------------------------
# Global constants and SPL templates
#---------------------------------------------------------------------------------

# IP Anomaly detection SPL
g_IP_Anomaly_Detection_SPL="""
| tstats summariesonly=t fillnull_value="---" c, first(fraud_web.Country) as Country, sum(fraud_web.bytes_in) as bytes_in_tot, sum(fraud_web.bytes_out) as bytes_out_tot, max(fraud_web.bytes_in) as bytes_in_max, max(fraud_web.bytes_out) as bytes_out_max
  FROM datamodel=fraud_web 
  WHERE {{TIMEFRAME}} NOT fraud_web.Country IN ("USA")
  BY fraud_web.src_ip fraud_web.http_user_agent fraud_web.http_method fraud_web.username_tried fraud_web.logged_in fraud_web.status fraud_web.uri_path
| rename fraud_web.* as *
| eval http_method=if(http_method="---", "Unknown", http_method)
| eval logged_in=if(NOT logged_in IN (0,1), "Unknown", logged_in)
| eval http_method__{http_method}=c
| eval logged_in__{logged_in}=c
| eval status__{status}=c
| eval username=username_tried
| fillnull
| eval ua_len=len(http_user_agent)

| stats sum(c) as ml__cnt, sum(bytes_in_tot) as bytes_in_tot, sum(bytes_out_tot) as bytes_out_tot, max(bytes_in_max) as ml__bytes_in_max, max(bytes_out_max) as ml__bytes_out_max, sum(bytes_in) as bytes_in, sum(bytes_out) as bytes_out, first(Country) as Country, dc(uri_path) as ml__pages, sum(logged_in__*) as ml__logged_in__*_c, sum(http_method__*) as ml__http_method__*_c, sum(status__*) as ml__status__*_c, dc(username) as ml__usernames, dc(http_user_agent) as ml__uas, min(ua_len) as ml__ua_len_min, max(ua_len) as ml__ua_len_max 
  BY src_ip
| eval ml__bytes_in_avg = round(bytes_in_tot/ml__cnt,1)
| eval ml__bytes_out_avg = round(bytes_out_tot/ml__cnt,1)
| foreach ml__*_c [eval <<FIELD>> = round((<<FIELD>>*100)/ml__cnt, 2)]

| fit StandardScaler ml__* with_mean=false with_std=true 
| fit PCA SS_* k=3
| rename mlx__* as ml__*
| fields - SS_* 
| eventstats median(PC_1) as centroid_PC_1, median(PC_2) as centroid_PC_2, median(PC_3) as centroid_PC_3
| eval v1=pow(PC_1 - centroid_PC_1, 2)
| eval v2=pow(PC_2 - centroid_PC_2, 2)
| eval v3=pow(PC_3 - centroid_PC_3, 2)
| eval res1 = sqrt(v1+v2+v3)
| eval anomaly_ratio = round(res1, 3)

| sort 0 -  anomaly_ratio

| streamstats c
| eval clusterId = case(src_ip IN ("170.0.236.103","173.212.199.29","176.31.115.122","190.129.24.154","190.129.35.236","190.129.35.238","190.129.35.246","192.99.144.140","198.27.80.140","201.0.207.40","62.141.45.203","85.230.196.115","85.230.196.55","85.230.198.102"), "PROPFIND", c<=5, "Anomaly", c<=10, "Warning", c<=20, "Suspicious", c<=30, "Outlier", true(), "Baseline")
| eval r=random()%100+1
| where clusterId!="Baseline" OR r<=5
| eval clusterColor = case(clusterId="PROPFIND", "#cb2196", clusterId="Anomaly", "#f91818", clusterId="Warning", "#fd7a35", clusterId="Suspicious", "#fbd049", clusterId="Outlier", "#b8d13d", true(), "#30c118")
| eval x=PC_1, y=PC_2, z=PC_3

| table clusterId x y z clusterColor anomaly_ratio src_ip Country centr* PC_* ml_*

```without it - dashboard freezes. ```
| table clusterId x y z clusterColor anomaly_ratio src_ip Country PC_*
| head 125
"""

g_User_Session_Anomaly_Detection_SPL="""
| from datamodel:fraud_web 
| search sourcetype=st_firstfederal {{TIMEFRAME}}
  action IN ("login_success", "login_fail", "edit_password","edit_profile","edit_username", "add_payee","money_movement","bill_payment","trade_securities") usernames!="*drye6*"
| eval x="Don't need super long sequences of the same thing"
| streamstats c by session_id action 
| where c<=20
| eval x="Set actions in chrono order"
| reverse
| eval x="Replace action names with a single char"
| eval action_short = case(action IN ("login_success"), "S", action IN ("login_fail"), "F", action IN ("edit_password","edit_profile","edit_username"), "E", action IN ("add_payee","money_movement","bill_payment","trade_securities"), "M", true(), null())

| stats first(usernames) as usernames, list(action_short) as actions1 by session_id

| eval actions1=mvjoin(actions1, "")
| eval x="Risky behavior == failed login OR at least 2 or more logins, followed by edit profile, followed by any money movement"
| eval risky=if(match(actions1, "F(F|(S.*?){2,}).*?E.*?M"), 1, 0)
| where risky=1
| eval risk_extra_F=3 * len(replace(actions1, "[^F]+", ""))
| eval risk_extra_E=2 * len(replace(actions1, "[^E]+", ""))
| eval risk_extra_M=4 * len(replace(actions1, "[^M]+", ""))
| eval username=split(usernames, "|")
| mvexpand username
| dedup session_id
| eval risk_score_total=50 + risk_extra_F + risk_extra_E + risk_extra_M
| eventstats values(session_id) as session_ids, values(username) as usernames
| sort 0 - risk_score_total
| fields - x risky usernames risk_extra* session_ids
"""
