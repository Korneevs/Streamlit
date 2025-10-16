import clickhouse_connect
import pandas as pd


class CHEngine():
    
    def __init__(
        self,
        user=None,
        clickhouse_password=None,
        port=None
    ):
        if port:
            self.client = clickhouse_connect.get_client(
                host='clickhouse-tcp-clickhouse-abcentral-production-rs-rs01.db.avito-sd',
                database='dwh',
                user=user,
                password=clickhouse_password,
                port=port,
            )
        else:
            self.client = clickhouse_connect.get_client(
                host='clickhouse-tcp-clickhouse-abcentral-production-rs-rs01.db.avito-sd',
                database='dwh',
                user=user,
                password=clickhouse_password,
                port=8123
            )
        
        
    def select(self, sql):
        result = self.client.query(sql)
        dataset = pd.DataFrame(result.result_rows, columns=result.column_names)
        return dataset