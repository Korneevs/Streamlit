import pandas as pd


class UtilityDataFetcher():
    
    
    def __init__(self):
        pass
    
    
    def create_ch_id_dicts(self, engine_c):

        regIds = engine_c.select("""SELECT * FROM dct.m42_region""")
        reg_to_id = pd.Series(regIds.id.values, index=regIds.value).to_dict()
        reg_to_id['Any'] = -1

        regIds = engine_c.select("""SELECT * FROM dct.m42_logical_category""")
        log_to_id = pd.Series(regIds.id.values, index=regIds.value).to_dict()
        log_to_id['Any'] = -1

        vertIds = engine_c.select("""SELECT * FROM dct.m42_vertical """)
        vert_to_id = pd.Series(vertIds.id.values, index=vertIds.value).to_dict()
        vert_to_id['Any'] = -1

        return vert_to_id, log_to_id, reg_to_id  
    
    
    def get_region_pops(self, engine_v):
        res = engine_v.select(
                '''
                WITH t1 AS 
                (
                SELECT Region, Population FROM dma.current_locations
                WHERE Level = 2 AND
                Country = 'все регионы' AND 
                Region NOT IN ('Undefined', 'другой регион', 'Москва и Московская область', 'Санкт-Петербург и Ленинградская область')
                ORDER BY Region
                )
                SELECT * FROM t1
                UNION ALL
                SELECT 'РФ' AS Region, SUM(Population) FROM t1
                '''
                )
        regions = dict(zip(res['Region'], res['Population']))
        return regions