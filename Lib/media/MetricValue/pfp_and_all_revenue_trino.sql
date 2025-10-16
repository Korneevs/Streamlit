select
    SUM(pfp_revenue) as pfp_revenue,
    SUM(total_revenue) as total_revenue
from
    dma.mrk_pfp_and_total_revenue
where True
    and event_date between DATE({calculation_start_date}) and DATE({calculation_end_date})
    and {vertical_condition}
    and {logical_category_condition}
    and {target_regions_condition}
