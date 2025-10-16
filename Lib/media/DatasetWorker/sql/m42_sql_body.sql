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
            argMax(metric_value, launch_id) AS metric_value
        FROM dma.m42 t
        WHERE t.metric_id IN ({metric_ids})
        AND is_human in (1)
        and metric_date >= '2020-01-01'
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
    dictGetOrDefault('dct.m42_region', 'value', toUInt64(region_id), 'Undefined') as region,
    dictGetOrDefault('dct.m42_city', 'value', toUInt64(city_id), 'Undefined') as city,
    {metric_columns}
--     dictGetOrDefault('dict.m42_metric', 'value', toUInt64(metric_id), 'Undefined') as metric,
--     SUM(metric_value) as metric_value
FROM base
GROUP BY metric_date, slice_name, group_name, region, city