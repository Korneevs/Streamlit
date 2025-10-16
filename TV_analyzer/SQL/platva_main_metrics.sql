SELECT
    pum.user_id AS user_id,
    pum.event_date::DATE AS dt,
    SUM(pum.buyers) AS buyers,
    SUM(pum.contacts) AS contacts,
    SUM(pum.dlu) AS DLU,
    SUM(pum.dtb) AS DTB,
    {other_metrics}
    SUM(pum.iv) AS iv
FROM
    dma.mrk_platva_user_metrics pum
WHERE True
    AND pum.logical_category_id IN (
        SELECT DISTINCT logical_category_id 
        FROM dma.current_logical_categories
        WHERE logical_category IN ({log_cat_str})
    )
    AND pum.event_date::DATE BETWEEN '{train_start_date}'::DATE AND '{analysed_end_date}'::DATE
GROUP BY
    1, 2
