SELECT
    adv_campaign AS flight_name,
    MIN(date_week_start - INTERVAL '21 days')::DATE AS train_start_date,
    MIN(date_week_start - INTERVAL '1 days')::DATE AS train_end_date,
    MIN(date_week_start)::DATE AS flight_start_date,
    MAX(date_week_finish + INTERVAL '6 days')::DATE AS flight_end_date,
    MIN(date_week_start)::DATE AS analysed_start_date,
    MAX(date_week_finish + INTERVAL '20 days')::DATE AS analysed_end_date,
    SUM(
        CASE
            WHEN budget_fact_amt = 0 THEN budget_plan_amt
            ELSE budget_fact_amt
        END
    ) AS flight_budget,
    LISTAGG(DISTINCT rk_ids USING PARAMETERS separator=', ', max_length=2048)::VARCHAR AS rk_clip_id,
    MAX(reach) AS reach
FROM dma.mrk_avito_flights_visualizer_report_data
WHERE channel ILIKE '%tv%'
  AND adv_campaign = '{flight_name_viz}'
  AND channel IN ('TV Sponsorship', 'Nat TV')
GROUP BY adv_campaign;
