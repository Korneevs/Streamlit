DROP TABLE IF EXISTS session_info;
CREATE LOCAL TEMP TABLE session_info ON COMMIT PRESERVE ROWS AS (
    SELECT
        *
    FROM (
        SELECT 
            CASE 
                WHEN dict.promo_vertical is NOT NULL THEN promo_vertical
                WHEN utmcampaign like '%auto%' THEN 'Transport'
                WHEN utmcampaign like '%\_re\_%' THEN 'Realty'
                WHEN utmcampaign like '%good%' THEN 'Goods'
                WHEN utmcampaign like '%services%' THEN 'Services'
                WHEN utmcampaign like '%jobs%' THEN 'Vacancies'
                ELSE NULL
            END AS promo_vertical,
            main.lnd_session_source_full_id,
            main.source_type_id
        FROM  DMA.session_source_full_dict as main
        left join DMA.current_mrk_started_campaign dict
            on main.utmcampaign = dict.utm_campaign        
    ) as main
    WHERE promo_vertical is not NULL
);


DROP TABLE IF EXISTS sessions_cookie_df;
CREATE LOCAL TEMP TABLE sessions_cookie_df ON COMMIT PRESERVE ROWS AS (
    select 
        ssf.session_start_dttm::DATE as start_date,
        session_source_full_id,
        source_type_id,
        cpt.cookie_id
    from dma.session_source_full ssf 
    INNER JOIN (
        SELECT DISTINCT cookie_id 
        FROM DMA.click_stream_contacts 
        WHERE EventDate::DATE >= '{start_date}'::DATE - INTERVAL '30 days'
        AND HASH(cookie_id) % 100 < {percent}
    ) as cpt
    USING(cookie_id)
    where session_start_dttm::date >= '{start_date}'::DATE
        and session_start_dttm::date <= '{end_date}' --:last_date
        and is_human = True
);


DROP TABLE IF EXISTS sessions_user_df;
CREATE LOCAL TEMP TABLE sessions_user_df ON COMMIT PRESERVE ROWS AS (
    select 
        start_date,
        session_source_full_id,
        source_type_id,
        user_id
    from sessions_cookie_df ssf 
    INNER JOIN (
        SELECT 
            DISTINCT
            cookie_id, user_id
        FROM DMA.mrk_cookie_user
    ) as cpt
    USING(cookie_id)
);


DROP TABLE IF EXISTS public.perf_contact_start_session;
CREATE TABLE public.perf_contact_start_session AS
/*+DIRECT*/
(
    select
        DISTINCT
        start_date,
        user_id,
        promo_vertical
    from sessions_user_df ssf 
    join /*+ JTYPE(FM) */ session_info as ssfd
        on ssf.session_source_full_id = ssfd.lnd_session_source_full_id
            and (ssf.source_type_id = ssfd.source_type_id )
    where promo_vertical is not NULL
) 
ORDER BY user_id
SEGMENTED BY hash(user_id) ALL NODES;
SELECT ANALYZE_STATISTICS('public.perf_contact_start_session');


