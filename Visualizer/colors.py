from enum import Enum

from plotly import express as px
import plotly.graph_objects as go
import plotly.io as pio


blue = px.colors.qualitative.T10[0]  # Таблошные цвета
grey = px.colors.qualitative.T10[9]
red = px.colors.qualitative.T10[2]
green = px.colors.qualitative.T10[4]
black = '#3d3d3d'
white = '#ffffff'
orange = '#ff6600'
yellow = '#ffff00'


axes_format = dict(
    tickfont_color='gray',
    tickfont_size=10,
    linecolor='lightgray',
    ticks='outside',
    tickcolor='lightgray',
    showline=True,
)

big10_theme = go.layout.Template(
    layout=go.Layout(
        template=pio.templates['plotly_white'],
        xaxis=dict(showgrid=False, tickformat="%b '%y", hoverformat='%d %b', **axes_format),
        yaxis=dict(showgrid=True, **axes_format),
    )
)