DROP TABLE IF EXISTS spillover_df;
CREATE LOCAL TEMP TABLE spillover_df ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(
    WITH start_session as (
        SELECT
            user_id,
            logical_category as start_logical_category,
            vertical as start_vertical,
            dt as start_date,
            dt + INTERVAL '{delta_period} days' as end_date,
            LAG(dt::DATE, 1) OVER (partition by user_id ORDER BY dt) as prev_date
        FROM public.ce_buyer_table
        WHERE 
            reg_date::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
        AND 
            dt >= '{start_date}'::DATE
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