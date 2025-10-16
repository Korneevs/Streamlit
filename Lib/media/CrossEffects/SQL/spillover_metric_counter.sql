DROP TABLE IF EXISTS spllover_start;
CREATE LOCAL TEMP TABLE spllover_start ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(
    WITH new_user_df as (
        SELECT
            user_id,
            RegistrationTime
        FROM DMA.current_user
        WHERE RegistrationTime::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
    ),
    main_df as (
        SELECT
            lc.logical_category,
            lc.vertical,
            csc.user_id AS user_id,
            csc.EventDate as curr_dt,
            DATEDIFF(day, '{start_date}'::DATE, RegistrationTime::DATE) as reg_ind,
            DATEDIFF(day, '{start_date}'::DATE, csc.EventDate) as day_ind,
            COUNT(1) as contact_num
        FROM DMA.click_stream_contacts AS csc
        INNER JOIN /*+ JTYPE(FM) */ INFOMODEL.current_infmquery_category i
            ON i.infmquery_id = csc.infmquery_id
        INNER JOIN /*+ JTYPE(FM) */ DMA.current_logical_categories lc
            ON lc.logcat_id = i.logcat_id
        INNER JOIN new_user_df
            ON csc.user_id = new_user_df.user_id
        WHERE TRUE
            AND csc.EventDate::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
            and csc.EventDate::DATE < '{end_date}'::DATE + INTERVAL '{delta_period} days'
            AND COALESCE(csc.ishuman, TRUE)
            AND RegistrationTime IS NOT NULL
            AND logical_category is not NULL
        GROUP BY 1, 2, 3, 4, 5, 6
    )
    SELECT
        logical_category,
        vertical,
        user_id,
        curr_dt,
        reg_ind,
        day_ind,
        contact_num,
        LAG(day_ind, 1) OVER (partition by user_id ORDER BY day_ind) as prev_ind,
        LEAD(day_ind, 1) OVER (partition by user_id ORDER BY day_ind) as next_ind
    FROM main_df
) 
ORDER BY user_id
SEGMENTED BY hash(user_id) ALL NODES;
SELECT ANALYZE_STATISTICS('spllover_start');


