/*
Параметры:
- channel,
- sql_channel_list,
- flights_list
*/

SELECT
    adv_campaign AS flight_name,
    vertical,
    logcat AS category,
    '{channel}' AS channel,
    tier,
    CASE
        WHEN SUM(CASE WHEN adv_type = 'Федеральная' THEN 1 ELSE 0 END) > 0
        THEN 'Federal'
        ELSE 'Regional'
    END AS adv_type,
    CASE
        WHEN SUM(
            CASE WHEN region LIKE '%РФ%' THEN 1 ELSE 0 END
        ) > 0
        THEN 'Any'
        ELSE array_join(array_agg(DISTINCT region), ', ')
    END AS region,
    CAST(MIN(date_week_start) AS DATE) AS date_start,
    CAST(MAX(date_week_finish) AS DATE) AS date_end,
    SUM(
        CASE
            WHEN budget_fact_amt = 0 THEN budget_plan_amt
            ELSE budget_fact_amt
        END
    ) AS budget
FROM
    dma.mrk_avito_flights_visualizer_report_data_trino
WHERE
    channel IN ({sql_channel_list})
    {extra_filter}
    AND constant = 1
GROUP BY
    adv_campaign,
    logcat,
    tier,
    vertical;