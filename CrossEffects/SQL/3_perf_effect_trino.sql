/*
Параметры:
- schema,
- delta_period.
*/

DROP TABLE IF EXISTS {schema}.perf_effect;
CREATE TABLE {schema}.perf_effect AS (
    SELECT
        vertical,
        main.promo_vertical,
        date_diff('day', main.start_date, dt) as day_ind,
        SUM(w_buyer_num) as w_buyer
    FROM public.perf_contact_start_session as main
    INNER JOIN public.ce_buyer_table as bt
    ON main.user_id = bt.user_id and dt >= main.start_date
        and dt < main.start_date + INTERVAL '{delta_period}' day
    GROUP BY 1, 2, 3
);