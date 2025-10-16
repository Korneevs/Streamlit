DROP TABLE IF EXISTS {table_name};

CREATE LOCAL TEMP TABLE {table_name} ON COMMIT PRESERVE ROWS AS
(
    -- Создаем полный список пользователей.
    WITH all_users AS (
        SELECT DISTINCT user_id FROM full_metric_table
        UNION 
        SELECT DISTINCT user_id FROM {tv_nontv_users_table}
    ),
    
    -- Создаем полный список дат.
    all_dates AS (
        SELECT DISTINCT dt FROM full_metric_table
    ),

    -- Делаем CROSS JOIN, создавая всевозможные пары (user_id, dt).
    all_users_dates AS (
        SELECT 
            u.user_id,
            d.dt
        FROM all_users AS u
        CROSS JOIN all_dates AS d
    )

    -- Подставляем метрики, если они есть, иначе заполняем нулями.
    SELECT
        aud.user_id AS user_id,
        COALESCE(tv_users."group", 'Avito') AS exp_group,
        aud.dt AS dt,
        {sum_metrics}
    FROM 
        all_users_dates AS aud
    LEFT JOIN 
        full_metric_table AS main
        ON aud.user_id = main.user_id 
        AND aud.dt = main.dt
    LEFT JOIN
        {tv_nontv_users_table} AS tv_users
        ON aud.user_id = tv_users.user_id
    GROUP BY 
        1, 2, 3
);
