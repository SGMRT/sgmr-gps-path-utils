import json
import pandas as pd
from typing import TypedDict, List
import plotly.express as px

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    return data


# Target data has these types
# { code: "OK", matchings: [], tracepoints: [{ location: [double, double] }] }
class Step(TypedDict):
    intersections: List[dict]

class Leg(TypedDict):
    steps: List[dict]

class Mathcing(TypedDict):
    confidence: float
    legs: List[dict]
    weight_name: str
    geometry: str
    weight: float
    duration: float
    distance: float

class Tracepoint(TypedDict):
    location: List[float]

class OriginalMapData(TypedDict):
    timestampe: str
    lat: float
    lng: float
    dist: float
    pace: float
    alt: int
    cadence: int
    bpm: int
    isRunning: bool

class OsrmMapData(TypedDict):
    code: str
    matchings: List[dict]
    tracepoints: List[Tracepoint]

# Non-OSRM data
before: List[OriginalMapData] = load_jsonl('dummy/data7.jsonl')
beforeDf = pd.DataFrame({
    "lon": [item["lng"] for item in before],
    "lat": [item["lat"] for item in before],
})

after: List[OriginalMapData] = load_jsonl('result/smoothed_data.jsonl')
afterDf = pd.DataFrame({
    "lon": [item["lng"] for item in after],
    "lat": [item["lat"] for item in after],
})

# OSRM data
data: OsrmMapData = load_json('result/osrm_result.json')
valid_tracepoints = [
    tp for tp in data['tracepoints']
    if tp is not None and 'location' in tp and isinstance(tp['location'], list) and len(tp['location']) == 2
]
osrmDf = pd.DataFrame({
    "lon": [tp['location'][0] for tp in valid_tracepoints],
    "lat": [tp['location'][1] for tp in valid_tracepoints],
})

# Single Scatter View
# scatter_fig = px.scatter_map(
#     beforeDf,
#     lat="lat",
#     lon="lon",
#     zoom=15,
#     height=1200,
#     map_style="open-street-map",
#     title="Tracepoints Map (Detailed Scatter)",
#     labels={"lat": "Latitude", "lon": "Longitude"},
#     hover_data=afterDf.columns,
#     color_discrete_sequence=["red"],
#     size_max=8
# )

# Multi Scatter View
scatter_fig = px.scatter_map(
    pd.concat([beforeDf.assign(type="Before"), afterDf.assign(type="After")]),
    lat="lat",
    lon="lon",
    zoom=15,
    height=1200,
    map_style="open-street-map",
    title="Tracepoints Map (Before vs After)",
    labels={"lat": "Latitude", "lon": "Longitude", "type": "Data Type"},
    hover_data=["lat", "lon", "type"],
    color="type",
    color_discrete_map={"Before": "red", "After": "blue"},
    size_max=8
)

# Single Line View
line_fig = px.line_map(
    afterDf,
    lat="lat",
    lon="lon",
    zoom=15,
    height=1200,
    map_style="open-street-map",
    title="Tracepoints Map (Detailed Line)",
    labels={"lat": "Latitude", "lon": "Longitude"},
    hover_data=afterDf.columns
)

# Mutiple Line View
# line_fig = px.line_map(
#     pd.concat([osrmDf.assign(type="OSRM"), afterDf.assign(type="After")]),
#     lat="lat",
#     lon="lon",
#     zoom=15,
#     height=1200,
#     map_style="open-street-map",
#     title="Tracepoints Map (OSRM vs After)",
#     labels={"lat": "Latitude", "lon": "Longitude", "type": "Data Type"},
#     hover_data=["lat", "lon", "type"],
#     color="type",
#     color_discrete_map={"OSRM": "red", "After": "blue"},
# )

scatter_fig.update_layout(mapbox_style="open-street-map")
scatter_fig.write_html("result/scatter_view.html")

line_fig.update_layout(mapbox_style="open-street-map")
line_fig.write_html("result/line_view.html")