DROP TABLE IF EXISTS public.ce_buyer_table;
CREATE TABLE public.ce_buyer_table AS
/*+DIRECT*/
(
    with users_to_observe as (
        SELECT 
            DISTINCT 
            user_id,
            RegistrationTime
        FROM DMA.current_user
        WHERE user_id in (SELECT user_id FROM public.perf_contact_start_session)
            OR
        RegistrationTime::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
    )
    SELECT
        DISTINCT
        lc.vertical,
        lc.logical_category,
        csc.user_id AS user_id,
        csc.EventDate::date as dt,
        RegistrationTime::DATE as reg_date,
        w.w as w_buyer_num
    FROM DMA.click_stream_contacts AS csc
    INNER JOIN users_to_observe as st
    on st.user_id = csc.user_id
    INNER JOIN /*+ JTYPE(FM) */ INFOMODEL.current_infmquery_category i
        ON i.infmquery_id = csc.infmquery_id
    INNER JOIN /*+ JTYPE(FM) */ DMA.current_logical_categories lc
        ON lc.logcat_id = i.logcat_id
    INNER JOIN weight_df as w
    USING (logical_category)
    WHERE TRUE
        AND csc.EventDate::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
        and csc.EventDate::DATE < '{end_date}'::DATE + INTERVAL '{delta_period} days'
        AND COALESCE(csc.ishuman, TRUE)
        AND lc.logical_category is not NULL
--     GROUP BY 1,2,3,4,5
)
ORDER BY user_id
SEGMENTED BY hash(user_id) ALL NODES;
SELECT ANALYZE_STATISTICS('public.ce_buyer_table');
