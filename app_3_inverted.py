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


def gen_sql_for_simple_mask(
    mask_type: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    if mask_type == 'time':
        sql = f'SELECT DISTINCT time'
    elif mask_type == 'pixel':
        sql = f'SELECT DISTINCT pid'
    else:
        sql = f'SELECT time, pid'
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    for variable, predicate, value in value_criteria:
        sql += f'\nAND {variable} {predicate} {value}'
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    if mask_type == 'time':
        sql += f'\nORDER BY time'
    print(sql)
    return sql


def gen_simple_mask(
    mask_type: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    sql = gen_sql_for_simple_mask(mask_type, start_time, end_time, shape, value_criteria)
    df = client.query_df(sql)
    print(df.head())
    return df


def gen_sql_for_agg_mask(
    mask_type: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    agg_value_criteria: list[tuple[str, str, str, float]] = None,
):
    agg_projection = ',\n'.join([f'{agg}If({variable}, isFinite({variable})) as {agg}_{short_2_long_map[variable]}' for agg, variable, _, _ in agg_value_criteria])
    if mask_type == 'time':
        sql = f'SELECT time, {agg_projection}'
    elif mask_type == 'pixel':
        sql = f'SELECT pid, {agg_projection}'
    else:
        raise ValueError('mask_type must be one of time, pixel')
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    if mask_type == 'time':
        sql += f'\nGROUP BY time'
    elif mask_type == 'pixel':
        sql += f'\nGROUP BY pid'
    else:
        raise ValueError('mask_type must be one of time, pixel')
    having = '\nAND '.join([f'{agg}If({variable}, isFinite({variable})) {predicate} {value}' for agg, variable, predicate, value in agg_value_criteria])
    sql += f'\nHAVING {having}'
    if mask_type == 'time':
        sql += f'\nORDER BY time'
    print(sql)
    return sql


def gen_agg_mask(
    mask_type: str,
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    agg_value_criteria: list[tuple[str, str, str, float]] = None,
):
    sql = gen_sql_for_agg_mask(mask_type, start_time, end_time, shape, agg_value_criteria)
    df = client.query_df(sql)
    print(df.head())
    return df


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
live_state = {}
# tab 1: by value
mk_type = pn.widgets.Select(options=['(time, pixel)', 'time', 'pixel'])
mk_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
mk_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
mk_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
mk_value_criteria = []
mk_criteria_variables = pn.widgets.Select(options=popular_variables)
mk_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
mk_input_val = pn.widgets.TextInput(value='0', width=100)
mk_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
mk_value_criteria_str = pn.pane.Str('Variable criteria:')
mk_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
mk_variable_mulit_choice = pn.widgets.MultiChoice(options=popular_variables)
mk_btn_mask_to_xarray = pn.widgets.Button(name='Mask to xarray', button_type='primary', width=100)
mk_plot_variable = pn.widgets.Select(options=[])
mk_btn_refresh_map = pn.widgets.Button(name='Refresh Map', button_type='primary', width=100)
mk_sidebar = pn.Column(
    '###### 1. Mask type',
    mk_type,
    '###### 2. Date range',
    mk_date_range,
    '###### 3. Region',
    mk_regions,
    mk_wkt_input,
    '###### 4. Variable value criteria',
    mk_criteria_variables,
    pn.Row(mk_predicates_val, mk_input_val, mk_btn_add_value_criteria),
    mk_value_criteria_str,
    '###### 5. Generate mask',
    mk_btn_run,
)


# botton on click
def mk_add_value_criteria(event):
    short = long_2_short_map[mk_criteria_variables.value]
    mk_value_criteria.append((short, mk_predicates_val.value, float(mk_input_val.value)))
    output = [f'{short_2_long_map[c[0]]} {c[1]} {c[2]}' for c in mk_value_criteria]
    mk_value_criteria_str.object = 'Variable criteria:'
    for num, i in enumerate(output):
        mk_value_criteria_str.object += f'\n {num + 1}. {i}'


def run_mask(event):
    mask_type = mk_type.value
    start_time, end_time = mk_date_range.value
    region_name = mk_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            input_wkt = mk_wkt_input.value
            shape = shapely.wkt.loads(input_wkt)
        else:
            shape = gdf_region[gdf_region['name'] == region_name].geometry.values[0]

    df = gen_simple_mask(
        mask_type=mask_type,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        value_criteria=mk_value_criteria,
    )
    mask_table.value = df.head(25)
    live_state['mask'] = df


mk_btn_add_value_criteria.on_click(mk_add_value_criteria)
mk_btn_run.on_click(run_mask)

# tab 2: by agg
ma_type = pn.widgets.Select(options=['time', 'pixel'])
ma_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
ma_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
ma_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
ma_value_criteria = []
ma_agg_type = pn.widgets.Select(options=['avg', 'min', 'max'])
ma_criteria_variables = pn.widgets.Select(options=popular_variables)
ma_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
ma_input_val = pn.widgets.TextInput(value='0', width=100)
ma_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
ma_value_criteria_str = pn.pane.Str('Aggregated value criteria:')
ma_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
ma_variable_mulit_choice = pn.widgets.MultiChoice(options=popular_variables)
ma_btn_mask_to_xarray = pn.widgets.Button(name='Mask to xarray', button_type='primary', width=100)
ma_plot_variable = pn.widgets.Select(options=[])
ma_btn_refresh_map = pn.widgets.Button(name='Refresh Map', button_type='primary', width=100)
ma_sidebar = pn.Column(
    '###### 1. Mask type',
    ma_type,
    '###### 2. Date range',
    ma_date_range,
    '###### 3. Region',
    ma_regions,
    ma_wkt_input,
    '###### 4. Aggregated value criteria',
    ma_agg_type,
    ma_criteria_variables,
    pn.Row(ma_predicates_val, ma_input_val, ma_btn_add_value_criteria),
    ma_value_criteria_str,
    '###### 5. Generate mask',
    ma_btn_run,
)


# botton on click
def ma_add_value_criteria(event):
    short = long_2_short_map[ma_criteria_variables.value]
    ma_value_criteria.append((ma_agg_type.value, short, ma_predicates_val.value, float(ma_input_val.value)))
    output = [f'{c[0]}({short_2_long_map[c[1]]}) {c[2]} {c[3]}' for c in ma_value_criteria]
    ma_value_criteria_str.object = 'Aggregated value criteria:'
    for num, i in enumerate(output):
        ma_value_criteria_str.object += f'\n {num + 1}. {i}'


def run_agg_mask(event):
    mask_type = ma_type.value
    start_time, end_time = ma_date_range.value
    region_name = ma_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            shape = shapely.wkt.loads(ma_wkt_input.value)
        else:
            shape = gdf_region[gdf_region.name == region_name].iloc[0].geometry

    df = gen_agg_mask(
        mask_type=mask_type,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        agg_value_criteria=ma_value_criteria,
    )
    mask_table.value = df.head(25)


ma_btn_add_value_criteria.on_click(ma_add_value_criteria)
ma_btn_run.on_click(run_agg_mask)
# END of components and behavior

# START of page layout
# sidebar layout
side_tabs = pn.WidgetBox(pn.Tabs(
    ('1. By Value', mk_sidebar),
    ('2. By Agg', ma_sidebar),
))

# main layout
mask_table = pn.widgets.Tabulator(None, show_index=False)
gv_image = (gf.ocean * gf.land * gf.coastline).opts(global_extent=True)
map_panel = pn.panel(gv_image, width=700, height=500)  # tab 3: map component
mk_variable_mulit_choice = pn.widgets.MultiChoice(options=popular_variables)
mk_btn_mask_to_xarray = pn.widgets.Button(name='Mask to xarray', button_type='primary', width=100)
xa_info = pn.pane.Str('Xarray info: ')  # tab 3: xarray info
main_tabs = pn.Tabs(
    ('1. Found time/pixel', mask_table),
    ('2. Mask to Array', pn.Column(
        pn.Row(
            '###### Mask Array',
            mk_variable_mulit_choice,
            mk_btn_mask_to_xarray,
        ),
        '###### Xarray info',
        xa_info,
    )),
)

# template
template = pn.template.BootstrapTemplate(
    title='IHARP ERA5 POC - Inverted Query',
    theme_toggle=False,
    sidebar=[side_tabs],
    main=[map_panel, main_tabs],
)
# END of page layout

template.servable()