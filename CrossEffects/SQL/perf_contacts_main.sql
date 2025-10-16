DROP TABLE IF EXISTS media_crosseffects_main;
CREATE LOCAL TEMP TABLE media_crosseffects_main ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(
    with weight as (
        select logical_category, 
        SUM(revenue) / SUM(buyers) as w
        from DMA.mrk_weights_by_quarters
        WHERE start_period::date >= '{start_date}'::DATE
        AND end_period::date < '{end_date}'::DATE + INTERVAL '{delta_period} days'
        GROUP BY 1
    ),
    main as (
        SELECT
            lc.vertical,
            promo_vertical,
            lc.logical_category,
            csc.user_id AS user_id,
            st.start_date,
            csc.EventDate as dt,
            st.start_date + INTERVAL '{delta_period} days' as end_date,
            DATEDIFF(day, st.start_date, csc.EventDate) as day_ind,
            w.w as w_buyer_num
        FROM DMA.click_stream_contacts AS csc
        INNER JOIN media_start_session as st
        on st.user_id = csc.user_id and csc.EventDate >= st.start_date 
            and csc.EventDate < st.start_date + INTERVAL '{delta_period} days'
        INNER JOIN /*+ JTYPE(FM) */ INFOMODEL.current_infmquery_category i
            ON i.infmquery_id = csc.infmquery_id
        INNER JOIN /*+ JTYPE(FM) */ DMA.current_logical_categories lc
            ON lc.logcat_id = i.logcat_id
        INNER JOIN weight as w
        USING (logical_category)
        WHERE TRUE
            AND csc.EventDate::DATE >= '{start_date}'::DATE
            and csc.EventDate::DATE < '{end_date}'::DATE + INTERVAL '{delta_period} days'
            AND COALESCE(csc.ishuman, TRUE)
            AND logical_category is not NULL
        GROUP BY 1, 2, 3, 4, 5, 6
    )
    SELECT
        vertical,
        promo_vertical,
        user_id,
        dt,
        start_date,
        end_date,
        day_ind,
        CASE WHEN prev_ind is NULL AND vertical = promo_vertical THEN 1 END as is_spillover,
        w_contact_num
    FROM (
        SELECT
            vertical,
            promo_vertical,
            user_id,
            dt,
            start_date,
            end_date,
            day_ind,
            LAG(day_ind, 1) OVER (partition by user_id, promo_vertical, start_date ORDER BY day_ind) as prev_ind,
            w_contact_num
        FROM main 
    ) as main
)
ORDER BY user_id
SEGMENTED BY hash(user_id) ALL NODES;
SELECT ANALYZE_STATISTICS('media_crosseffects_main');
