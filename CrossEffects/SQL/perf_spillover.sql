DROP TABLE IF EXISTS media_perf_spillover;
CREATE LOCAL TEMP TABLE media_perf_spillover ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(
    with weight as (
        select logical_category,
        SUM(revenue) / SUM(buyers) as w
        from DMA.mrk_weights_by_quarters
        WHERE start_period::date >= '{start_date}'::DATE
        AND end_period::date < '{end_date}'::DATE + INTERVAL '{delta_period} days'
        GROUP BY 1
    )
    SELECT
        DISTINCT
        lc.vertical,
        promo_vertical,
        csc.user_id AS user_id,
        st.start_date,
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
)
ORDER BY user_id
SEGMENTED BY hash(user_id) ALL NODES;
SELECT ANALYZE_STATISTICS('media_perf_spillover');
