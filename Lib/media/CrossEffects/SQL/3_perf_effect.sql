DROP TABLE IF EXISTS perf_effect;
CREATE LOCAL TEMP TABLE perf_effect ON COMMIT PRESERVE ROWS AS (
    SELECT
        vertical,
        main.promo_vertical,
        DATEDIFF(day, main.start_date, dt) as day_ind,
        SUM(w_buyer_num) as w_buyer
    FROM public.perf_contact_start_session as main
    INNER JOIN public.ce_buyer_table as bt
    ON main.user_id = bt.user_id and dt >= main.start_date
        and dt < main.start_date + INTERVAL '{delta_period} days'
    GROUP BY 1, 2, 3
)