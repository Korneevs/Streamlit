/*
Параметры:
- fligh_name
*/

SELECT
    adv_campaign AS flight_name,
    SUM(
        CASE
            WHEN budget_fact_amt = 0 THEN budget_plan_amt
            ELSE budget_fact_amt
        END
    ) AS flight_budget
FROM
    dma.mrk_avito_flights_visualizer_report_data_trino
WHERE constant = 1
    {extra_filter}
GROUP BY
    adv_campaign;