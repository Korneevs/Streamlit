/*
Параметры:
- table_name
*/

SELECT
  flight_name,
  vertical,
  category,
  channel,
  date_start,
  date_end,
  analysed_date_start,
  analysed_date_end,
  tier,
  type,
  flight_budget,
  channel_budget,
  channel_budget_analysed,
  analysis_tool,
  main_metric,
  is_main_metric,
  proxy_metric,
  rel_metric_uplift,
  rel_metric_mde,
  abs_metric_uplift,
  abs_metric_mde,
  romi,
  mde_romi,
  revenue,
  metric_to_revenue,
  metric_to_revenue_elasticity,
  cross_effect,
  long_term_effect,
  sum_metric_per_flight,
  cost_per_metric,
  verdict,
  main_log_cats,
  cf_link,
  custom_slice_name_1,
  custom_slice_uplift_1,
  custom_slice_mde_1,
  custom_slice_name_2,
  custom_slice_uplift_2,
  custom_slice_mde_2,
  custom_slice_name_3,
  custom_slice_uplift_3,
  custom_slice_mde_3,
  custom_slice_name_4,
  custom_slice_uplift_4,
  custom_slice_mde_4
FROM (
    SELECT
        t.*,
        ROW_NUMBER() OVER (
            PARTITION BY flight_name
            ORDER BY               
                version_stamp DESC, 
                id            DESC  
        ) AS rn
    FROM {table_name} AS t  
    WHERE is_main_metric = true
) sub
WHERE rn = 1;