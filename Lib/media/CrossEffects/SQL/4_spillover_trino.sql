/*
Параметры:
- schema,
- start_date,
- delta_period.
*/

DROP TABLE IF EXISTS {schema}.spillover_df;
CREATE TABLE {schema}.spillover_df AS
(
    WITH start_session as (
        SELECT
            user_id,
            logical_category as start_logical_category,
            vertical as start_vertical,
            dt as start_date,
            dt + INTERVAL '{delta_period}' day as end_date,
            LAG(DATE(dt), 1) OVER (partition by user_id ORDER BY dt) as prev_date
        FROM public.ce_buyer_table
        WHERE 
            DATE(reg_date) >= DATE('{start_date}') - INTERVAL '30' day
        AND 
            dt >= DATE('{start_date}')
    )
    SELECT
        bt.{slice_name},
        start_{slice_name},
        SUM(w_buyer_num) as w_buyer
    FROM start_session as main
    INNER JOIN public.ce_buyer_table as bt
    ON main.user_id = bt.user_id and dt >= main.start_date
        and dt < main.end_date
    WHERE prev_date is NULL
    GROUP BY 1, 2
);