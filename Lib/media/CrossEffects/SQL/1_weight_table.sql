DROP TABLE IF EXISTS weight_df;
CREATE LOCAL TEMP TABLE weight_df ON COMMIT PRESERVE ROWS AS
/*+DIRECT*/
(        
    select 
        logical_category, 
        SUM(revenue) / SUM(buyers) as w
    from DMA.mrk_weights_by_quarters
    WHERE start_period::date >= '{start_date}'::DATE
    AND end_period::date < '{end_date}'::DATE + INTERVAL '{delta_period} days'
    GROUP BY 1
)