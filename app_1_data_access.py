import pickle
from datetime import date, datetime

import clickhouse_connect
import geopandas as gpd
import geoviews as gv
import geoviews.feature as gf
import numpy as np
import pandas as pd
import panel as pn
import shapely

# START of query method
client = clickhouse_connect.get_client(password='huan1531')


def get_pid_str(shape):
    gdf_sjoin = gdf_pid.sjoin(gpd.GeoDataFrame({'geometry': [shape]}, crs=4326), how='inner', predicate='within')
    pids_str = ", ".join(np.char.mod('%d', gdf_sjoin['pid'].values))
    return pids_str


def xarray_from_clickhouse_pid(
    variables: list[str],
    start_time: datetime = None,
    end_time: datetime = None,
    shape: shapely.Geometry = None,
    value_criteria: list[tuple[str, str, float]] = [],
):
    sql = f'SELECT time, longitude, latitude, {", ".join(variables)}'
    sql += f'\nFROM pop_pid'
    if start_time is not None and end_time is not None:
        sql += f"\nWHERE time BETWEEN '{start_time}' AND '{end_time}'"
    for variable, predicate, value in value_criteria:
        sql += f'\nAND {variable} {predicate} {value}'
    if shape is not None:
        pids_str = get_pid_str(shape)
        sql += f'\nAND pid IN ({pids_str})'
    print(sql)

    # query to df
    df = client.query_df(sql)
    print(df.head())

    # df to xarray
    df = df.set_index(['time', 'longitude', 'latitude'])
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

live_state = {}

# START of components and behavior

# sidebar components
da_variable_mulit_choice = pn.widgets.MultiChoice(options=popular_variables)
da_date_range = pn.widgets.DatetimeRangePicker(start=date(2022, 5, 1), end=date(2022, 5, 7), value=(datetime(2022, 5, 1, 0, 0, 0), datetime(2022, 5, 1, 23, 59, 59)))
da_regions = pn.widgets.Select(options=['Whole World', 'greenland', 'iceland', 'tri', 'rec', 'wkt'])
da_wkt_input = pn.widgets.TextAreaInput(value='POLYGON((-72 78,-13 81,-44 60,-72 78))')
da_value_criteria = []
da_criteria_variables = pn.widgets.Select(options=popular_variables)
da_predicates_val = pn.widgets.Select(options=['>', '>=', '<', '<=', '=', '<>'], width=50)
da_input_val = pn.widgets.TextInput(value='0', width=100)
da_btn_add_value_criteria = pn.widgets.Button(name='Add', button_type='primary', width=100)
da_btn_erase_value_criteria = pn.widgets.Button(name='Erase', button_type='primary', width=100)
da_value_criteria_str = pn.pane.Str('Variable criteria:')
da_btn_run = pn.widgets.Button(name='Run', button_type='primary', width=100)
da_plot_variable = pn.widgets.Select(options=[])
da_btn_refresh_map = pn.widgets.Button(name='Refresh Map', button_type='primary', width=100)
da_sidebar = pn.Column(
    '###### 1. Variables',
    da_variable_mulit_choice,
    '###### 2. Date range',
    da_date_range,
    '###### 3. Region',
    da_regions,
    da_wkt_input,
    '###### 4. Variable value criteria',
    da_criteria_variables,
    pn.Row(da_predicates_val, da_input_val),
    pn.Row(da_btn_add_value_criteria, da_btn_erase_value_criteria),
    da_value_criteria_str,
    da_btn_run,
    '###### 5. Refresh map',
    da_plot_variable,
    da_btn_refresh_map,
)


# botton on click
def add_value_criteria(event):
    short = long_2_short_map[da_criteria_variables.value]
    da_value_criteria.append((short, da_predicates_val.value, float(da_input_val.value)))
    output = [f'{short_2_long_map[c[0]]} {c[1]} {c[2]}' for c in da_value_criteria]
    da_value_criteria_str.object = 'Variable criteria:'
    for num, i in enumerate(output):
        da_value_criteria_str.object += f'\n {num + 1}. {i}'


def get_xarray(event):
    da_plot_variable.options = da_variable_mulit_choice.value
    da_data_vars_short = [long_2_short_map[v] for v in da_variable_mulit_choice.value]
    start_time, end_time = da_date_range.value
    region_name = da_regions.value
    shape = None
    if region_name != 'Whole World':
        if region_name == 'wkt':
            input_wkt = da_wkt_input.value
            shape = shapely.wkt.loads(input_wkt)
        else:
            shape = gdf_region[gdf_region['name'] == region_name].geometry.values[0]
    ds = xarray_from_clickhouse_pid(
        variables=da_data_vars_short,
        start_time=start_time,
        end_time=end_time,
        shape=shape,
        value_criteria=da_value_criteria,
    )
    live_state['ds'] = ds
    xa_info.object = f'{ds}'
    xa_table.value = ds.to_dataframe().reset_index().head(10)

    # update aggregation result
    agg = []
    for short in da_data_vars_short:
        long = short_2_long_map[short]
        ds_min = ds[ds[short].argmin(...)]
        min_val = ds_min[short].values
        min_lon = ds_min['longitude'].values
        min_lat = ds_min['latitude'].values
        min_time = ds_min['time'].values
        ds_max = ds[ds[short].argmax(...)]
        max_val = ds_max[short].values
        max_lon = ds_max['longitude'].values
        max_lat = ds_max['latitude'].values
        max_time = ds_max['time'].values
        mean = ds.mean()[short].values
        agg.append((long, min_val, min_lon, min_lat, min_time, max_val, max_lon, max_lat, max_time, mean))
    df = pd.DataFrame.from_records(agg, columns=['Variable', 'Min', 'Lon of Min', 'Lat of Min', 'Time of Min', 'Max', 'Lon of Max', 'Lat of Max', 'Time of Max', 'Mean'])
    xa_agg.value = df


def refresh_map(event):
    # update map
    ds = live_state['ds']
    viz_short = long_2_short_map[da_plot_variable.value]
    gv_ds = gv.Dataset(ds, vdims=viz_short)
    image = gv_ds.to(gv.Image, ['longitude', 'latitude'])
    image.opts(cmap='coolwarm', colorbar=True, tools=['hover'])
    image = image * gf.coastline
    map_panel.object = image
    print('Done updating map!!')


def erase_value_criteria(event):
    da_value_criteria_str.object = 'Variable criteria:'
    da_value_criteria.clear()


da_btn_add_value_criteria.on_click(add_value_criteria)
da_btn_run.on_click(get_xarray)
da_btn_refresh_map.on_click(refresh_map)
da_btn_erase_value_criteria.on_click(erase_value_criteria)

# END of components and behavior

# START of page layout

# map component
gv_image = (gf.ocean * gf.land * gf.coastline).opts(global_extent=True)
map_panel = pn.panel(gv_image, width=700, height=500)

# xarray page
xa_info = pn.pane.Str('Xarray info: ')
xa_table = pn.widgets.Tabulator(None, show_index=False)
xa_agg = pn.widgets.Tabulator(None, show_index=False)
xa_page = pn.Column(
    '##### 1. Xarray Info',
    xa_info,
    '##### 2. Xarray as Table',
    xa_table,
    '##### 3. Xarray Aggregation',
    xa_agg,
)

# template
template = pn.template.BootstrapTemplate(
    title='IHARP ERA5 POC - Data Access',
    theme_toggle=False,
    sidebar=[da_sidebar],
    main=[
        map_panel,
        xa_page,
    ],
)

# END of page layout

template.servable()