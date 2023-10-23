import pickle
from datetime import date, datetime

import clickhouse_connect
import geopandas as gpd
import geoviews as gv
import geoviews.feature as gf
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import shapely

# START of query method
client = clickhouse_connect.get_client(password='huan1531')


def get_pid_str(shape):
    gdf_sjoin = gdf_pid.sjoin(gpd.GeoDataFrame({'geometry': [shape]}, crs=4326), how='inner', predicate='within')
    pids_str = ", ".join(np.char.mod('%d', gdf_sjoin['pid'].values))
    return pids_str


def agg_from_clickhouse_pid(
    mma: str,
    variable: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    sql = f'SELECT {mma}({variable}) AS {mma}_{variable}'
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    for variable, predicate, value in value_criteria:
        sql += f'\nAND {variable} {predicate} {value}'
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    print(sql)

    res = client.query(sql)
    agg = res.first_row[0]
    print(agg)
    return agg


def groupby_time_from_clickhouse_pid(
    variables: list[str],
    time_unit: str,
    agg: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    if time_unit not in ['year', 'month', 'week', 'day', 'hour']:
        raise ValueError('time_unit must be one of year, month, week, day, hour')
    time_trunc = f"date_trunc('{time_unit}', time) "
    agg_str = ', '.join([f'{agg}If({variable}, isFinite({variable})) AS {agg}_{short_2_long_map[variable]}' for variable in variables])
    sql = f'SELECT {time_trunc} AS {time_unit}, {agg_str}'
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    for variable, predicate, value in value_criteria:
        sql += f'\nAND {variable} {predicate} {value}'
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    sql += f'\nGROUP BY {time_trunc}'
    sql += f'\nORDER BY {time_trunc}'
    print(sql)

    df = client.query_df(sql)
    df = df.set_index(time_unit)
    print(df.head())
    return df


def groupby_pid_from_clickhouse_pid(
    variables: list[str],
    agg: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    agg_str = ', '.join([f'{agg}If({variable}, isFinite({variable})) AS {agg}_{variable}' for variable in variables])
    sql = f'SELECT longitude, latitude, {agg_str}'
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    for variable, predicate, value in value_criteria:
        sql += f'\nAND {variable} {predicate} {value}'
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    sql += f'\nGROUP BY longitude, latitude'
    print(sql)

    # query to df
    df = client.query_df(sql)
    print(df.head())

    # df to xarray
    df = df.set_index(['longitude', 'latitude'])
    ds = df.to_xarray()
    print(ds)
    return ds


# END of query method

# START of preload data
popular_variables = pickle.load(open("data/pickle/popular_variable.pkl", "rb"))
long_2_short_map = pickle.load(open("data/pickle/file_variable_map.pkl", "rb"))
short_2_long_map = {v: k for k, v in long_2_short_map.items()}

gdf_list = []
for r in ['greenland', 'iceland', 'tri', 'rec']:
    gdf = gpd.read_file(f'data/vector/{r}.geojson')
    gdf['name'] = r
    gdf_list.append(gdf)
gdf_region = pd.concat(gdf_list, ignore_index=True)
gdf_region = gdf_region.set_crs(4326)

df_pid = pd.read_csv('data/vector/pid.csv')
gdf_pid = gpd.GeoDataFrame(df_pid, geometry=gpd.points_from_xy(x=df_pid.longitude, y=df_pid.latitude, crs=4326))
# END of preload data

# START of components and behavior
# tab 1: single value
agg_method = pn.widgets.Select(options=['avg', 'min', 'max'])
agg_variable_select = pn.widgets.Select(options=popular_variables)
agg_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
agg_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
agg_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
agg_value_criteria = []
agg_criteria_variables = pn.widgets.Select(options=popular_variables)
agg_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
agg_input_val = pn.widgets.TextInput(value='0', width=100)
agg_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
agg_value_criteria_str = pn.pane.Str('Variable criteria:')
agg_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
agg_sidebar = pn.Column(
    '###### 1. Aggregation method',
    agg_method,
    '###### 2. Variables',
    agg_variable_select,
    '###### 3. Date range',
    agg_date_range,
    '###### 4. Region',
    agg_regions,
    agg_wkt_input,
    '###### 5. Variable value criteria',
    agg_criteria_variables,
    pn.Row(agg_predicates_val, agg_input_val, agg_btn_add_value_criteria),
    agg_value_criteria_str,
    agg_btn_run,
)


# botton on click
def agg_add_value_criteria(event):
    short = long_2_short_map[agg_criteria_variables.value]
    agg_value_criteria.append((short, agg_predicates_val.value, float(agg_input_val.value)))
    output = [f'{short_2_long_map[c[0]]} {c[1]} {c[2]}' for c in agg_value_criteria]
    agg_value_criteria_str.object = 'Variable criteria:'
    for num, i in enumerate(output):
        agg_value_criteria_str.object += f'\n {num + 1}. {i}'


def run_agg(event):
    mma = agg_method.value
    variable = long_2_short_map[agg_variable_select.value]
    start_time, end_time = agg_date_range.value
    region_name = agg_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            input_wkt = agg_wkt_input.value
            shape = shapely.wkt.loads(input_wkt)
        else:
            shape = gdf_region[gdf_region['name'] == region_name].geometry.values[0]

    res = agg_from_clickhouse_pid(
        mma=mma,
        variable=variable,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        value_criteria=agg_value_criteria,
    )

    agg_single_value_result.object = f'###### {mma} {agg_variable_select.value}: {res}'


agg_btn_add_value_criteria.on_click(agg_add_value_criteria)
agg_btn_run.on_click(run_agg)

# tab 2: time series
ts_variable_mulit_choice = pn.widgets.MultiChoice(options=popular_variables)
ts_time_unit = pn.widgets.Select(options=['hour', 'day', 'week', 'month', 'year'])
ts_agg_method = pn.widgets.Select(options=['avg', 'min', 'max'])
ts_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
ts_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
ts_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
ts_value_criteria = []
ts_criteria_variables = pn.widgets.Select(options=popular_variables)
ts_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
ts_input_val = pn.widgets.TextInput(value='0', width=100)
ts_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
ts_value_criteria_str = pn.pane.Str('Variable criteria:')
ts_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
ts_sidebar = pn.Column(
    '###### 1. Variables',
    ts_variable_mulit_choice,
    '###### 2. Time unit',
    ts_time_unit,
    '###### 3. Aggregation method',
    ts_agg_method,
    '###### 4. Date range',
    ts_date_range,
    '###### 5. Region',
    ts_regions,
    ts_wkt_input,
    '###### 6. Variable value criteria',
    ts_criteria_variables,
    pn.Row(ts_predicates_val, ts_input_val, ts_btn_add_value_criteria),
    ts_value_criteria_str,
    ts_btn_run,
)


# botton on click
def ts_add_value_criteria(event):
    short = long_2_short_map[ts_criteria_variables.value]
    ts_value_criteria.append((short, ts_predicates_val.value, float(ts_input_val.value)))
    output = [f'{short_2_long_map[c[0]]} {c[1]} {c[2]}' for c in ts_value_criteria]
    ts_value_criteria_str.object = 'Variable criteria:'
    for num, i in enumerate(output):
        ts_value_criteria_str.object += f'\n {num + 1}. {i}'


def run_ts(event):
    variables = [f'{long_2_short_map[v]}' for v in ts_variable_mulit_choice.value]
    time_unit = ts_time_unit.value
    agg_method = ts_agg_method.value
    start_time, end_time = ts_date_range.value
    region_name = ts_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            input_wkt = ts_wkt_input.value
            shape = shapely.wkt.loads(input_wkt)
        else:
            shape = gdf_region[gdf_region['name'] == region_name].geometry.values[0]

    df = groupby_time_from_clickhouse_pid(
        variables=variables,
        time_unit=time_unit,
        agg=agg_method,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        value_criteria=ts_value_criteria,
    )

    fig = px.line(df)
    ts_plotly_plot.object = fig


ts_btn_add_value_criteria.on_click(ts_add_value_criteria)
ts_btn_run.on_click(run_ts)

# tab 3: one map
live_state = {}
om_agg_method = pn.widgets.Select(options=['avg', 'min', 'max'])
om_agg_variable_multi_choise = pn.widgets.MultiChoice(options=popular_variables)
om_agg_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
om_agg_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
om_agg_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
om_agg_value_criteria = []
om_agg_criteria_variables = pn.widgets.Select(options=popular_variables)
om_agg_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
om_agg_input_val = pn.widgets.TextInput(value='0', width=100)
om_agg_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
om_agg_value_criteria_str = pn.pane.Str('Variable criteria:')
om_agg_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
om_plot_variable = pn.widgets.Select(options=[])
om_btn_refresh_map = pn.widgets.Button(name='Refresh Map', button_type='primary', width=100)
om_agg_sidebar = pn.Column(
    '###### 1. Aggregation method',
    om_agg_method,
    '###### 2. Variables',
    om_agg_variable_multi_choise,
    '###### 3. Date range',
    om_agg_date_range,
    '###### 4. Region',
    om_agg_regions,
    om_agg_wkt_input,
    '###### 5. Variable value criteria',
    om_agg_criteria_variables,
    pn.Row(om_agg_predicates_val, om_agg_input_val, om_agg_btn_add_value_criteria),
    om_agg_value_criteria_str,
    om_agg_btn_run,
    '###### 6. Plot',
    om_plot_variable,
    om_btn_refresh_map,
)


# botton on click
def om_add_value_criteria(event):
    short = long_2_short_map[om_agg_criteria_variables.value]
    om_agg_value_criteria.append((short, om_agg_predicates_val.value, float(om_agg_input_val.value)))
    output = [f'{short_2_long_map[c[0]]} {c[1]} {c[2]}' for c in om_agg_value_criteria]
    om_agg_value_criteria_str.object = 'Variable criteria:'
    for num, i in enumerate(output):
        om_agg_value_criteria_str.object += f'\n {num + 1}. {i}'


def run_om_agg(event):
    mma = om_agg_method.value
    short_vars = [long_2_short_map[v] for v in om_agg_variable_multi_choise.value]
    start_time, end_time = om_agg_date_range.value
    region_name = om_agg_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            input_wkt = om_agg_wkt_input.value
            shape = shapely.wkt.loads(input_wkt)
        else:
            shape = gdf_region[gdf_region['name'] == region_name].geometry.values[0]

    ds = groupby_pid_from_clickhouse_pid(
        variables=short_vars,
        agg=mma,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        value_criteria=om_agg_value_criteria,
    )

    live_state['ds'] = ds
    xa_info.object = f'{ds}'
    om_plot_variable.options = om_agg_variable_multi_choise.value


def refresh_map(event):
    ds = live_state['ds']
    viz_short = long_2_short_map[om_plot_variable.value]
    gv_ds = gv.Dataset(ds, vdims=f'{om_agg_method.value}_{viz_short}')
    image = gv_ds.to(gv.Image, ['longitude', 'latitude'])
    image.opts(cmap='coolwarm', colorbar=True, tools=['hover'])
    image = image * gf.coastline
    map_panel.object = image


om_agg_btn_add_value_criteria.on_click(om_add_value_criteria)
om_agg_btn_run.on_click(run_om_agg)
om_btn_refresh_map.on_click(refresh_map)
# END of components and behavior

# START of page layout
# sidebar layout
side_tabs = pn.WidgetBox(pn.Tabs(
    ('1. Single Value', agg_sidebar),
    ('2. Time Series', ts_sidebar),
    ('3. One Map', om_agg_sidebar),
))

# main layout
agg_single_value_result = pn.pane.Markdown('###### Aggregation result:')  # tab 1
ts_plotly_plot = pn.pane.Plotly(None)  # tab 2
gv_image = (gf.ocean * gf.land * gf.coastline).opts(global_extent=True)
map_panel = pn.panel(gv_image, width=700, height=500)  # tab 3: map component
xa_info = pn.pane.Str('Xarray info: ')  # tab 3: xarray info
main_tabs = pn.Tabs(
    ('1. Single Value', agg_single_value_result),
    ('2. Time Series', ts_plotly_plot),
    ('3. One Map', pn.Column(map_panel, xa_info)),
)

# template
template = pn.template.BootstrapTemplate(
    title='IHARP ERA5 POC - Direct Aggregation',
    theme_toggle=False,
    sidebar=[side_tabs],
    main=[main_tabs],
)
# END of page layout

template.servable()