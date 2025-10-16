DROP TABLE IF EXISTS spillover_df;
CREATE LOCAL TEMP TABLE spillover_df ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(
    with weight as (
        select 
        CASE 
            WHEN '{slice_name}' = 'vertical' THEN split_part(logical_category, '.', 1)
            ELSE logical_category
        END as {slice_name}, 
        SUM(revenue) / SUM(buyers) as w
        from DMA.mrk_weights_by_quarters
        WHERE start_period::date >= '{start_date}'::DATE
        AND end_period::date < '{end_date}'::DATE + INTERVAL '{delta_period} days'
        GROUP BY 1
    ),
    start_session as (
        SELECT
            DISTINCT
            user_id,
            curr_dt,
            day_ind as start_ind,
            day_ind + {delta_period} as end_ind,
            logical_category as start_logical_category,
            vertical as start_vertical
        FROM spllover_start
        WHERE start_ind < {delta_period} AND start_ind >= 0 AND prev_ind is NULL AND reg_ind is NOT NULL
    ),
    promo_start_session as (
        SELECT
            DISTINCT
            main.user_id,
            perf_table.promo_vertical,
            main.curr_dt as promo_start_date
        FROM spllover_start as main
        LEFT JOIN media_crosseffects_main as perf_table
            ON main.user_id = perf_table.user_id and main.curr_dt >= perf_table.start_date
        WHERE main.day_ind < {delta_period} AND main.day_ind >= 0 AND prev_ind is NULL AND reg_ind is NOT NULL
        AND perf_table.promo_vertical IS NOT NULL
    ),
    one_start as (
       SELECT
            main.user_id,
            perf_table.promo_vertical,
            start_logical_category,
            start_vertical,
            start_ind,
            end_ind
        FROM start_session as main
        LEFT JOIN promo_start_session as perf_table
            ON main.user_id = perf_table.user_id and main.curr_dt >= perf_table.promo_start_date
    )
    SELECT
        main.{slice_name},
        start_{slice_name},
        promo_vertical,
        SUM(contact_num * w) as w_contact_num
    FROM spllover_start as main
    INNER JOIN weight as w
        USING ({slice_name})
    INNER JOIN one_start as st
        ON main.user_id = st.user_id and main.day_ind >= st.start_ind and main.day_ind <= st.end_ind
    GROUP BY 1, 2, 3
);