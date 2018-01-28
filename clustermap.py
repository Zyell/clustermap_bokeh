from bokeh.layouts import gridplot, column
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.models import Spacer, CustomJS, DataRange1d, ColumnDataSource, \
    LinearColorMapper, ColorBar, BasicTicker, Div, Tool, Column, HoverTool
from bokeh.transform import transform
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Union, Dict


def clustermap(df: pd.DataFrame, title: Optional[str] = '',
               toolbar_location: Optional[str] = 'right', cmap: Optional[List[str]] = palettes.viridis(256),
               dendogram_method: Optional[str] = 'average', dendogram_width: Optional[int] = 100,
               row_dendogram_location: Optional[str] = 'left', col_dendogram_location: Optional[str] = 'above',
               colorbar_location: Optional[str] = 'below', heatmap_size: Optional[Tuple[int, int]] = (600, 600),
               heatmap_x_axis_location: Optional[str] = 'below', heatmap_y_axis_location: Optional[str] = 'right',
               heatmap_x_col: Optional[str] = 'level_1', heatmap_y_col: Optional[str] = 'level_0',
               heatmap_select_by: Optional[str] = 'column', heatmap_desired_num_ticks: Optional[int] = 7,
               heatmap_measure_label: Optional[str] = '',) -> Column:

    """

    Generate a seaborn inspired clusteramp (heatmap & dendograms)

    :param df: dataframe of data
    :param title: string title
    :param toolbar_location: placement of bokeh toolbar, options 'above', 'below', 'left', 'right'
    :param cmap: list of colors as string hex values
    :param dendogram_method: method to use for dendogram calculation in scipy cluster hiearchy
    :param dendogram_width: width of dendogram in pixels off of the heatmap plot
    :param row_dendogram_location: placement of row dendogram, options 'left', 'right'
    :param col_dendogram_location: placement of column dendogram, options 'above', 'below'
    :param colorbar_location: placement of heatmap colorbar, options 'above', 'below', 'left', 'right'
    :param heatmap_size: a tuple of integers delcaring with and height of heatmap in pixels
    :param heatmap_x_axis_location: location of x axis on heatmap, options 'above', 'below'
    :param heatmap_y_axis_location: location of y axis on heatmap, options 'left', 'right'
    :param heatmap_x_col: df column name that will be plotted along x axis of heatmap
    :param heatmap_y_col: df column name that will be plotted along y axis of heatmap
    :param heatmap_select_by: selection criteria, when tap tool is engages, options 'row' or 'column'
    :param heatmap_desired_num_ticks: desired number of tick marks on the heatmap colorbar
    :param heatmap_measure_label: label for tooltip of color mapped value
    :return: bokeh column layout of final clustermap
    """

    if row_dendogram_location not in ('left', 'right'):
        raise ValueError('row_dendogram_location must be in (left, right)')
    if col_dendogram_location not in ('above', 'below'):
        raise ValueError('col_dendogram_location must be in (above, below)')
    rows_linkage = hierarchy.linkage(df.values, method=dendogram_method)
    cols_linkage = hierarchy.linkage(df.T, method=dendogram_method)
    rows_dendogram = hierarchy.dendrogram(rows_linkage, no_plot=True)
    cols_dendogram = hierarchy.dendrogram(cols_linkage, no_plot=True)
    r_p = build_dendogram_plot(rows_dendogram, *get_dendogram_bounds(rows_dendogram, df.shape[0]),
                               width=dendogram_width, height=heatmap_size[1],
                               location=row_dendogram_location)
    c_p = build_dendogram_plot(cols_dendogram, *get_dendogram_bounds(cols_dendogram, df.shape[1]),
                               width=heatmap_size[0], height=dendogram_width,
                               location=col_dendogram_location)
    cols = list(df.columns)
    h_p = build_heatmap_plot(df,
                             [cols[c] for c in cols_dendogram['leaves']],
                             list(reversed(rows_dendogram['ivl'])),
                             cmap=cmap, colorbar_location=colorbar_location,
                             x_axis_location=heatmap_x_axis_location,
                             y_axis_location=heatmap_y_axis_location,
                             desired_num_ticks=heatmap_desired_num_ticks,
                             x_col=heatmap_x_col, y_col=heatmap_y_col,
                             select_by=heatmap_select_by)

    hover = h_p.select(HoverTool)
    hover.tooltips = [(f"{heatmap_x_col}", f"@{heatmap_x_col}"),
                      (f"{heatmap_y_col}", f"@{heatmap_y_col}"),
                      (f"{heatmap_measure_label}", "@measure")]

    h_p.min_border_left = 0
    h_p.min_border_top = 0
    r_p.min_border = 0
    c_p.min_border = 0

    callback_x = CustomJS(args=dict(x=c_p.x_range), code="""
    var start = cb_obj.start;
    var end = cb_obj.end;
    x.start = start;
    x.end = end;
        """)

    callback_y = CustomJS(args=dict(y=r_p.y_range), code=f"""
    var start = cb_obj.start;
    var end = cb_obj.end;
    y.end = {df.shape[0]} - end;
    y.start = {df.shape[0]} - start;
        """)

    h_p.x_range.callback = callback_x
    h_p.y_range.callback = callback_y
    c_p.outline_line_color = None
    r_p.outline_line_color = None

    if col_dendogram_location == 'above' and row_dendogram_location == 'left':
        gp = gridplot([[Spacer(width=dendogram_width), c_p], [r_p, h_p]],
                      toolbar_location=toolbar_location)
    elif col_dendogram_location == 'below' and row_dendogram_location == 'left':
        gp = gridplot([[r_p, h_p], [Spacer(width=dendogram_width), c_p]],
                      toolbar_location=toolbar_location)
    elif col_dendogram_location == 'above' and row_dendogram_location == 'right':
        gp = gridplot([[c_p, Spacer(width=dendogram_width)], [h_p, r_p]],
                      toolbar_location=toolbar_location)
    elif col_dendogram_location == 'below' and row_dendogram_location == 'right':
        gp = gridplot([[h_p, r_p], [c_p, Spacer(width=dendogram_width)]],
                      toolbar_location=toolbar_location)
    else:
        gp = None
    return column(Div(text=title), gp)


def build_heatmap_plot(data: pd.DataFrame, x_range: List[str], y_range: List[str], title: Optional[str] = '',
                       size: Optional[Tuple[int]] = (600, 600),
                       x_col: Optional[str] = 'level_1', y_col: Optional[str] = 'level_0',
                       select_by: Optional[str] = 'column', cmap: Optional[List[str]] = palettes.viridis(256),
                       tools: Optional[Union[str, List[Tool]]] = 'pan,box_zoom,tap,hover,reset',
                       x_axis_location: Optional[str] = 'below', y_axis_location: Optional[str] = 'right',
                       colorbar_location: Optional[str] = 'below', desired_num_ticks: Optional[int] = 7) -> figure:

    """

    heatmap building function

    :param data: dataframe containing heatmap data that will subsequently be stacked
    :param x_range: list of string factors
    :param y_range: list of string factors
    :param title: string name of plot
    :param size: tuple of integers delcaring width and height of plot in pixels
    :param x_col: column label of x data in stacked dataframe
    :param y_col: column label of y data in stacked dataframe
    :param select_by: selection criteria, when tap tool is engages, options 'row' or 'column'
    :param cmap: list of colors as string hex values
    :param tools: string of listed plot tools or List of bokeh tool instances
    :param x_axis_location: location of x axis, options 'above', 'below'
    :param y_axis_location: location of y axis, options 'left', 'right'
    :param colorbar_location: placement of colorbar, options 'above', 'below', 'left', 'right'
    :param desired_num_ticks: desired number of tick marks on the colorbar
    :return: bokeh figure object representing heatmap
    """

    if select_by not in ('column', 'row'):
        raise ValueError('select_by value must be in (column, row)')
    df = pd.DataFrame(data.stack(), columns=['measure']).reset_index()
    df[x_col] = df[x_col].astype(str)
    df[y_col] = df[y_col].astype(str)

    source = ColumnDataSource(df)

    col_to_select = x_col if select_by == 'column' else y_col
    selection_callback = CustomJS(args=dict(source=source), code=f"""
    console.log(source.data);
    var selection = cb_obj.selected['1d']['indices'][0];
    var label_selection = source.data['{col_to_select}'][selection];
    var selected = [];
    var labels = source.data['{col_to_select}'];
    for (i=0; i<labels.length; i++) {{
        if (labels[i] == label_selection) {{
            selected.push(i);
        }}
    }}
    source.selected['1d']['indices'] = selected;
    source.change.emit();
        """)

    source.js_on_change('selected', selection_callback)

    mapper = LinearColorMapper(palette=cmap, low=df.measure.min(), high=df.measure.max())

    plt = figure(plot_width=size[0], plot_height=size[1],
                 y_range=y_range, x_range=x_range,
                 toolbar_location=None, title=title,
                 tools=tools, x_axis_location=x_axis_location,
                 y_axis_location=y_axis_location)
    plt.rect(x=x_col, y=y_col, width=1, height=1, source=source,
             line_color=transform('measure', mapper),
             fill_color=transform('measure', mapper))

    orientation = 'horizontal' if colorbar_location in ('above', 'below') else 'vertical'
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=desired_num_ticks),
                         orientation=orientation)

    plt.add_layout(color_bar, colorbar_location)

    plt.x_range.bounds = 'auto'
    plt.y_range.bounds = 'auto'
    plt.axis.axis_line_color = None
    plt.axis.major_tick_line_color = None
    plt.axis.major_label_text_font_size = "5pt"
    plt.axis.major_label_standoff = 0
    plt.xaxis.major_label_orientation = 1.0
    return plt


def get_dendogram_bounds(dg: Dict, data_range: int) -> Tuple[int, int]:

    """

    function to calculate bounds and reduction factor for plotting dendograms

    :param dg: dictionary output from scipy dendogram calculation
    :param data_range: length of data in dendogram clustering direction
    :return: the max data length in clustering direction and reduction factor to match dendogram
        calculation to data
    """

    mins, maxs = zip(*((min(cls), max(cls)) for cls in dg['icoord']))
    dg_max = min(mins) + max(maxs)
    reduction_factor = dg_max / data_range
    return dg_max, reduction_factor


def build_dendogram_plot(dg: Dict, dg_max: int, reduction_factor: int, title: Optional[str] = '',
                         location: Optional[str] = 'above',
                         width: Optional[int] = 600, height: Optional[int] = 100) -> figure:

    """

    build dendogram figure

    :param dg: dictionary output from scipy dendogram calculation
    :param dg_max: max data length in direction of clustering
    :param reduction_factor: factor to match dendogram calculation to data
    :param title: string title of plot
    :param location: dendogram's eventual location for alignment along a heatmap
    :param width: width of dendogram in pixels
    :param height: height of dendogram in pixels
    :return: bokeh figure representing dendogram
    """

    if location == 'above':
        y_axis_location = 'left'
        x_axis_location = 'above'
        y_axis_flip = False
        x_axis_flip = False
        xs = [np.array(c) / reduction_factor for c in dg['icoord']]
        ys = dg['dcoord']
        start, end = 0, dg_max
        x_range = DataRange1d(start=start/reduction_factor,
                              end=end/reduction_factor)
        y_range = DataRange1d()
    elif location == 'left':
        y_axis_location = 'left'
        x_axis_location = 'below'
        y_axis_flip = True
        x_axis_flip = True
        ys = [np.array(c) / reduction_factor for c in dg['icoord']]
        xs = dg['dcoord']
        end, start = 0, dg_max
        x_range = DataRange1d()
        y_range = DataRange1d(start=start/reduction_factor,
                              end=end/reduction_factor)
    elif location == 'right':
        y_axis_location = 'right'
        x_axis_location = 'below'
        y_axis_flip = True
        x_axis_flip = False
        ys = [np.array(c) / reduction_factor for c in dg['icoord']]
        xs = dg['dcoord']
        end, start = 0, dg_max
        x_range = DataRange1d()
        y_range = DataRange1d(start=start/reduction_factor,
                              end=end/reduction_factor)
    elif location == 'below':
        y_axis_location = 'left'
        x_axis_location = 'below'
        y_axis_flip = True
        x_axis_flip = False
        xs = [np.array(c) / reduction_factor for c in dg['icoord']]
        ys = dg['dcoord']
        start, end = 0, dg_max
        x_range = DataRange1d(start=start/reduction_factor,
                              end=end/reduction_factor)
        y_range = DataRange1d()
    else:
        raise ValueError('location must be one of 4 options (above, left, right, below)')

    p = figure(title=title, height=height, width=width,
               toolbar_location=None, tools='reset', x_range=x_range, y_range=y_range,
               y_axis_location=y_axis_location, x_axis_location=x_axis_location)
    p.axis.visible = False
    p.grid.visible = False
    p.x_range.flipped = x_axis_flip
    p.y_range.flipped = y_axis_flip
    p.multi_line(xs, ys)
    return p


if __name__ == "__main__":
    iris = sns.load_dataset("iris")
    species = iris.pop("species")
    output_file("clustermap.html")
    show(clustermap(iris, title='<h3>Iris Data Clustermap</h3>'))
