with base AS (
    SELECT
        metric_date,
        {slice_column} AS slice_name,
        {group_column} AS group_name, 
        metric_id,
        region_id,
        city_id,
        -- if(group_name == 'possible control', region_id, -2) as region,
        -- это костыль для дедубликации строк,
        -- когда по какой-то причине одинаковые данные приезжают на разные шарды
        -- hopefully дубли будут пофикшены, @asfilatov ищет причину
        sum(metric_value) AS metric_value
    FROM (
        SELECT
--             _host_name,
            metric_date AS metric_date, 
            {all_slices_no_launch_id},
            max_by(metric_value, launch_id) AS metric_value
        FROM ab.dma.v_m42_v1 t
        WHERE t.metric_id IN ({metric_ids})
        AND is_human in (1)
        AND is_participant_new  in (0,1)
        AND metric_date >= date'2020-01-01'
        {conditions}
        GROUP BY 
--         _host_name, 
        metric_date, {all_slices_no_launch_id}
    ) t
    GROUP BY metric_date, {all_slices_no_launch_id}
)
SELECT 
    metric_date,
    slice_name,
    group_name,
    region_id,
    city_id,
    {metric_columns}
--     dictGetOrDefault('dict.m42_metric', 'value', toUInt64(metric_id), 'Undefined') as metric,
--     SUM(metric_value) as metric_value
FROM base
GROUP BY metric_date, slice_name, group_name, region_id, city_id