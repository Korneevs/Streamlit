final_sql_dict = {
    # Метрики пользователей по типу группы: Тест / Контроль.
    'user_group_df': """
        SELECT 
            user_id,
            exp_group,
            {user_metrics}
        FROM {name}
        WHERE exp_group != 'Avito'
        GROUP BY 1, 2
    """,
    # Случайная подвыборка Авито для оценки репрезентативности.
    'avito_sample_df': """
        SELECT 
            user_id,
            exp_group,
            {user_metrics}
        FROM {name}
        WHERE exp_group = 'Avito'
        GROUP BY 1, 2
        ORDER BY RANDOM()
        LIMIT 1000000
    """,
    # Временной ряд в разрезе каждой из групп: Тест / Контроль / Авито.
    'date_df': """
        SELECT 
            exp_group,
            dt,
            {date_metrics}
        FROM {name}
        GROUP BY 1, 2
    """
}