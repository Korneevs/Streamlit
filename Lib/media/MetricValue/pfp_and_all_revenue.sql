select 
    sum(case when 
        product_type = 'vas' and product_subtype = 'profile promo' 
        or product_type = 'cpt' 
        or product_type = 'cpa' 
        or product_type = 'promo' and transaction_subtype = 'trx promo' 
        or product_type = 'short_term_rent' and transaction_subtype = 'buyer book'
        then transaction_amount_net_adj
        else 0
        end) as pfp_revenue,
    sum(transaction_amount_net_adj) as total_revenue
from (
        with
        current_transaction_type as
        (
            select transactiontype_id, transaction_type, transaction_subtype, product_type, product_subtype, IsRevenue as is_revenue
            from DMA.current_transaction_type ctt
        )
        select
            ur.user_id,
            ur.event_date,
            ur.location_id,
            l.Region as region,
            lc.vertical,
            lc.logical_category,
            ctt.transaction_type,
            ctt.transaction_subtype,
            ctt.product_subtype,
            ctt.product_type,
            ctt.is_revenue,
            ur.transaction_amount_net_adj
        from DMA.paying_user_report ur
        join current_transaction_type ctt on ctt.transactiontype_id = ur.transactiontype_id
        left join DMA.current_locations l on l.Location_id = ur.location_id and ur.location_id != -1
        left join infomodel.current_infmquery_category i
                on i.infmquery_id = ur.infmquery_id and ur.infmquery_id != -1
        left join dma.current_logical_categories lc on lc.logcat_id = i.logcat_id
        where ur.user_id not in (select cu.user_id from dma."current_user" cu where cu.IsTest)
            and cast(ur.event_date as date) between {calculation_start_date} and {calculation_end_date}
            and {vertical_condition}
            and {logical_category_condition}
            and {target_regions_condition}
            and is_revenue = True
            and (product_type = 'vas' and product_subtype = 'profile promo' 
                or product_type = 'cpt' 
                or product_type = 'cpa' 
                or product_type = 'vas'
                or product_type = 'lf' 
                or product_type = 'promo' and transaction_subtype = 'trx promo' 
                or product_type = 'short_term_rent' and transaction_subtype = 'buyer book' 
                or product_type = 'subscription' and transaction_subtype = 'tariff lf package')
    ) init;