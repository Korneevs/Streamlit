from dataclasses import dataclass


@dataclass
class MetaInfo():
    metrics: str
    start_date: str
    end_date: str
    flight_end_date: str
    label: str