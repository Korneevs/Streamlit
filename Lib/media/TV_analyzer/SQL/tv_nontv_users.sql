/*
Параметры:
- date_week_start,
- date_week_end,
- full_period,
- rk_ids,
- channels.
*/

-- Таблица с флагами.
DROP TABLE IF EXISTS tv_flags;
CREATE LOCAL TEMP TABLE tv_flags ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        hid,
        MIN(tv_set_top_box_flag) AS tv_set_top_box_flag,
        MIN(rt_with_wink_flag) AS rt_with_wink_flag,
        MIN(wink_flag) AS wink_flag
    FROM dma.mrk_platva_tv_flags
    WHERE date_month BETWEEN DATE_TRUNC('month', '{date_week_start}'::DATE)
                          AND DATE_TRUNC('month', '{date_week_end}'::DATE)
    GROUP BY hid
);

-- Таблица со связками HID <-> USER_ID, а также среднее количество логов в месяц.
DROP TABLE IF EXISTS full_user_hid_table;
CREATE LOCAL TEMP TABLE full_user_hid_table ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        hu.user_id::INT AS user_id,
        hu.hid AS hid,
        ROUND(SUM(hu.cnt_hid_user_id) / COUNT(DISTINCT hu.date_month), 2) AS cnt_hid_user_id_by_month
    FROM dma.mrk_platva_hid_user_id AS hu
    JOIN tv_flags AS tf ON hu.hid = tf.hid
    WHERE hu.date_month <= DATE_TRUNC('month', '{date_week_end}'::DATE)
      AND tf.tv_set_top_box_flag
      AND tf.rt_with_wink_flag
      AND tf.wink_flag
      -- Условие на черный список для hid.
      AND hu.hid NOT IN (
              SELECT external_id AS hid
              FROM DDS.H_HID 
              WHERE HID_id IN (SELECT DISTINCT HID_id FROM DDS.S_HID_Blacklisted)
      )
    GROUP BY hu.user_id, hu.hid
);

-- Словарь для таблицы dma.mrk_platva_tv_type_channel_contact_hours.
DROP TABLE IF EXISTS dict_platva_tv_type_channel_contact_hours;
CREATE LOCAL TEMP TABLE dict_platva_tv_type_channel_contact_hours ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        t1.TVChannel_id AS channel_name_type_id,
        t1.s_tvchannel_name AS channel_name,
        t2.s_tvchannel_type AS type
    FROM (
        SELECT
            TVChannel_id,
            Name AS s_tvchannel_name
        FROM (
            SELECT
                TVChannel_id,
                Name,
                ROW_NUMBER() OVER (PARTITION BY TVChannel_id ORDER BY actual_date DESC) AS rn
            FROM DDS.S_TVChannel_Name
        ) t
        WHERE rn = 1
    ) t1
    JOIN (
        SELECT
            TVChannel_id,
            "Type" AS s_tvchannel_type
        FROM (
            SELECT
                TVChannel_id,
                "Type",
                ROW_NUMBER() OVER (PARTITION BY TVChannel_id ORDER BY actual_date DESC) AS rn
            FROM DDS.S_TVChannel_Type
        ) t
        WHERE rn = 1
    ) t2 ON t1.TVChannel_id = t2.TVChannel_id
);

-- Таблица с общими просмотрами ТВ.
DROP TABLE IF EXISTS watching_tv_table;
CREATE LOCAL TEMP TABLE watching_tv_table ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        t.hid,
        SUM(CASE WHEN dt.channel_name IN ({channels}) THEN tv_contact_hours ELSE 0 END) / {full_period} AS avito_rk_channel_watching,
        SUM(CASE WHEN dt.type = 'archive' THEN tv_contact_hours ELSE 0 END) / {full_period} AS archive_watching,
        SUM(CASE WHEN dt.type = 'archive' AND dt.channel_name IN ({channels}) THEN tv_contact_hours ELSE 0 END) / {full_period} AS archive_avito_rk_watching,
        SUM(tv_contact_hours) / {full_period} AS tv_watching,
        COUNT(DISTINCT t.channel_name_type_id) AS channel_cnt
    FROM dma.mrk_platva_tv_type_channel_contact_hours t
    JOIN dict_platva_tv_type_channel_contact_hours dt ON t.channel_name_type_id = dt.channel_name_type_id
    WHERE t.date_week_start BETWEEN '{date_week_start}'::DATE AND '{date_week_end}'::DATE
    GROUP BY t.hid
);

-- Таблица с просмотрами рекламных роликов.
DROP TABLE IF EXISTS tv_commercial_full;
CREATE LOCAL TEMP TABLE tv_commercial_full ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        hid,
        commercial,
        cnt AS tv_avito_contacts
    FROM dma.mrk_platva_tv_commercials
    WHERE date_week_start BETWEEN '{date_week_start}'::DATE AND '{date_week_end}'::DATE
);

-- Таблица со средними просмотрами в домохозяйстве в день по каналам.
DROP TABLE IF EXISTS tv_channel_debug_table;
CREATE LOCAL TEMP TABLE tv_channel_debug_table ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        t.hid,
        dt.channel_name AS channel,
        SUM(tv_contact_hours) / {full_period} AS tv_hours_watched_per_day
    FROM dma.mrk_platva_tv_type_channel_contact_hours t
    JOIN dict_platva_tv_type_channel_contact_hours dt ON t.channel_name_type_id = dt.channel_name_type_id
    WHERE t.date_week_start BETWEEN '{date_week_start}'::DATE AND '{date_week_end}'::DATE
    GROUP BY t.hid, dt.channel_name
);

-- Таблица с HID <-> USER_ID и группой (TEST / CONTROL).
DROP TABLE IF EXISTS tv_nontv_users;
CREATE LOCAL TEMP TABLE tv_nontv_users ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    WITH tv_table AS (
        SELECT
            main.hid,
            CASE WHEN tv_cnt_num = 0 THEN 0 ELSE 1 END AS tv_contact
        FROM (
            SELECT
                main.hid,
                archive_avito_rk_watching,
                SUM(
                    CASE
                        WHEN commercial IN ({rk_ids}) THEN tv_avito_contacts
                        ELSE 0
                    END
                ) AS tv_cnt_num
            FROM watching_tv_table main
            LEFT JOIN tv_commercial_full USING (hid)
            GROUP BY main.hid, archive_avito_rk_watching
        ) AS main
        JOIN full_user_hid_table AS add_info USING (hid)
        WHERE (tv_cnt_num = 0 AND archive_avito_rk_watching = 0)
           OR (tv_cnt_num > 0)
    )
    SELECT
        hid,
        user_id,
        CASE WHEN raw_group = 1 THEN 'test' ELSE 'control' END AS "group"
    FROM (
        SELECT
            main.user_id,
            MAX(CASE WHEN tv_contact IS NOT NULL THEN tv_contact ELSE 0 END) AS raw_group,
            MAX(main.hid) AS hid
        FROM full_user_hid_table AS main
        LEFT JOIN tv_table AS tv_seen USING (hid)
        WHERE tv_contact IS NOT NULL OR main.hid NOT IN (SELECT DISTINCT hid FROM watching_tv_table)
        GROUP BY main.user_id
    ) AS main
);

-- Таблица со средними просмотрами каналов в минутах для каждой группы (TEST / CONTROL).
DROP TABLE IF EXISTS channel_in_group_watchings;
CREATE LOCAL TEMP TABLE channel_in_group_watchings ON COMMIT PRESERVE ROWS AS /*+DIRECT*/ (
    SELECT
        main."group",
        channel,
        SUM(tv_hours_watched_per_day) / all_hids * 60 AS avg_watching_time_in_minutes
    FROM (
        SELECT DISTINCT "group", hid
        FROM tv_nontv_users
    ) AS main
    JOIN tv_channel_debug_table AS tv ON main.hid = tv.hid
    JOIN (
        SELECT "group", COUNT(DISTINCT hid) AS all_hids
        FROM tv_nontv_users
        GROUP BY "group"
    ) AS all_hids ON all_hids."group" = main."group"
    GROUP BY main."group", channel, all_hids.all_hids
);