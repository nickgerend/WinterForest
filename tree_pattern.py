# Written by: Nick Gerend, @dataoutsider
# Viz: "Winter Forest", enjoy!

import pandas as pd
import numpy as np
import os
from math import pi, exp, sqrt, log10, sin, cos
import random
import matplotlib.pyplot as plt

class point:
    def __init__(self, index, item, x, y, path, row, column, entity, tree = -1, w_t = -1, h_t = -1, w_k = -1, h_k = -1, year = -1): 
        self.index = index
        self.item = item
        self.x = x
        self.y = y
        self.path = path
        self.row = row
        self.column = column
        self.entity = entity
        self.tree = tree
        self.w_t = w_t
        self.h_t = h_t
        self.w_k = w_k
        self.h_k = h_k
        self.year = year
    def to_dict(self):
        return {
            'index' : self.index,
            'item' : self.item,
            'x' : self.x,
            'y' : self.y,
            'path' : self.path,
            'row' : self.row,
            'column' : self.column,
            'entity' : self.entity,
            'tree' : self.tree,
            'w_t' : self.w_t,
            'h_t' : self.h_t,
            'w_k' : self.w_k,
            'h_k' : self.h_k,
            'year' : self.year }

#region functions
def sigmoid_xy(x1, y1, x2, y2, points, orientation = 'h', limit = 6):
    x_1 = x1
    y_1 = y1
    x_2 = x2
    y_2 = y2
    if orientation == 'v':
        x1 = y_1
        y1 = x_1
        x2 = y_2
        y2 = x_2
    x = []
    y = []
    amin = 1./(1.+exp(limit))
    amax = 1./(1.+exp(-limit))
    da = amax-amin
    for i in range(points):
        i += 1
        xi = (i-1.)*((2.*limit)/(points-1.))-limit
        yi = ((1.0/(1.0+exp(-xi)))-amin)/da
        x.append((xi-(-limit))/(2.*limit)*(x2-x1)+x1)
        y.append((yi-(0.))/(1.)*(y2-y1)+y1)
    return { 'h': list(zip(x,y)), 'v': list(zip(y,x))}.get(orientation, None)

def ornament_xy(width, height, quad_points, limit = 4.):
    SW = sigmoid_xy(0., 0., -width/2., height/2., quad_points, 'v', limit)
    NW = sigmoid_xy(-width/2., height/2, 0., height, quad_points, 'v', limit)  
    NE = sigmoid_xy(0., height,width/2, height/2,  quad_points, 'v', limit)
    SE = sigmoid_xy(width/2, height/2, 0., 0., quad_points, 'v', limit)
    list_xy = SW + NW + NE + SE
    return list_xy

def lattice_xy(rows, columns, quad_points, limit = 4., width = 1., height = 2., tree = -1, w_t = -1, h_t = -1):
    r = rows
    c = columns
    list_o_xy = ornament_xy(width, height, quad_points, limit)
    x_shift = 0.
    y_shift = -2.0
    list_xy = []
    ix = 0
    item = 1
    for i in range(columns+2):
        for j in range(rows+1):
            for k in range(len(list_o_xy)):
                list_xy.append(point(ix, item, list_o_xy[k][0]+x_shift, list_o_xy[k][1]+y_shift, k, j, i, 'lattice', tree, w_t, h_t))
            x_shift += width
            item += 1
        if i % 2 == 0:
            x_shift = 0.5
            rows -= 1
        else:
            x_shift = 0.
            rows += 1
        y_shift += height/2
    list_xy_lattice = [i for i in list_xy if (i.x >= 0.) and (i.x <= r)and (i.y >= 0.) and (i.y <= c)]
    #list_xy_lattice = [i for i in list_xy]
    return list_xy_lattice

def x_curve(x, max_x):
    half_x = max_x/2.
    # if x == 0 or x == half_x or x == max_x:
    #     return x
    rem = 0
    if x <= half_x:
        rem = x
    else:
        rem = max_x - x
    x_i = rem/half_x
    #x_curve = sqrt(1.-x_i**2.)
    x_curve = 1.-x_i**2
    x_scale = half_x*x_curve
    if x <= half_x:
        return half_x-x_scale
    else:
        return half_x+x_scale

def x_transform(new_min, new_max, x, min_x, max_x):
    return (new_max-new_min)*(x-min_x)/(max_x-min_x)+new_min

def tree_knots_unit_matrix(rows, columns, knot_count, spot_min, points, radius = 0.5, tree = -1, w_t = -1, h_t = -1, data = None):
    if points % 2 != 0:
        points += 1
    spot_max = (rows-1)*(columns-1)
    rs = random.sample(range(10, spot_max-spot_min), knot_count)
    list_xy = []
    angle = 0.
    ix = 1
    x_shift = 0.
    r_list = []
    v_list = []
    t_list = []
    if data is not None:
        for i, row in data.iterrows():
            r_list.append(row['Area (hectares)'])
            v_list.append(row['Number of seedlings'])
            t_list.append(row['Year'])
    for i in range(len(rs)):
        index = rs[i]
        # left to right fill:
        check = index % columns
        if check == 0:
            xo = columns-1
        else:
            xo = index % (columns-1)
        yo = int(index/columns)+1
        if yo % 2 == 0:
            x_shift = 0.5
        else:
            x_shift = 0.
        # draw a circle for each index:
        if data is not None:
            radius = 0.3*(r_list[i]/data['Area (hectares)'].max())
        for j in range(points+1):
            xc = radius*sin(angle*pi/180.)
            yc = radius*cos(angle*pi/180.)
            if data is not None:
                list_xy.append(point(ix, i, xc+xo+x_shift, yc+yo, j, yo, xo, 'knot', tree, w_t, h_t, r_list[i], v_list[i], t_list[i]))
            else:
                list_xy.append(point(ix, i, xc+xo+x_shift, yc+yo, j, yo, xo, 'knot'))          
            angle += 1./points*360.
            ix += 1        
    return list_xy
#endregion

#region test 1
# x = np.linspace(0., 11., num=21)
# y = np.linspace(0., 0., num=21)
# x_t = [x_curve(i, 11.) for i in x]
# plt.scatter(x_t, y)
# plt.show()
#endregion

#region test 2
# rows = 10
# columns = 10
# list_xy_lattice = lattice_xy(rows, columns, 31)
# list_xy_lattice2 = tree_knots_unit_matrix(rows, columns, 12, 30, 50, 0.3)
# list_xy_lattice = list_xy_lattice + list_xy_lattice2
# # list_x = [i.x for i in list_xy_lattice]
# # list_y = [i.y for i in list_xy_lattice]
# # plt.scatter(list_x, list_y)
# # plt.show()
# df_out = pd.DataFrame.from_records([s.to_dict() for s in list_xy_lattice])
# df_out['x'] = [x_curve(i, columns) for i in df_out['x']]
# df_out.to_csv(os.path.dirname(__file__) + '/lattice.csv', encoding='utf-8', index=False)
#endregion

#region input
df_area = pd.read_csv(os.path.dirname(__file__) + '/NFD - Area planted by ownership and species group - EN FR.csv', engine = 'python')
df_seeds = pd.read_csv(os.path.dirname(__file__) + '/NFD - Number of seedlings planted by ownership, species group - EN FR.csv', engine = 'python')
df_area = df_area.loc[df_area['Area (hectares)'] > 0]
df_seeds = df_seeds.loc[df_seeds['Number of seedlings'] > 0]
df_combo = new_df = pd.merge(df_seeds, df_area,  how='left', left_on=['Jurisdiction','Species group', 'Year', 'Tenure (En)'], right_on = ['Jurisdiction','Species group', 'Year', 'Tenure (En)'])
df_combo = df_combo[['Year', 'Jurisdiction', 'Species group', 'Tenure (En)', 'Number of seedlings', 'Area (hectares)', 'Data qualifier', 'Data Qualifier']]
df_combo.rename(columns={'Data qualifier': 'Qualifier Seeds', 'Data Qualifier': 'Qualifier Area', 'Species group': 'Species', 'Tenure (En)': 'Tenure'}, inplace=True)
df_combo = df_combo.dropna()
print(df_combo)
#endregion

#region algorithm
forest = []
combo_group = df_combo.groupby(['Jurisdiction','Species', 'Tenure'], as_index=False)
buffer = 2.
base_buffer = 2.
lattice_points = 7
knot_points = 15

tree_i = 1
base_height = 20 #10
base_width = 10
list_xy_i = []
list_xy = []
tree_shift_x = buffer
test = 0
for name, row in combo_group:
    width = log10(row['Area (hectares)'].sum())
    height = log10(row['Number of seedlings'].sum())
    year_count = row['Year'].count()
    rows = base_height-int(height) #+
    columns = base_width
    lattice = lattice_xy(columns, rows, lattice_points, 4., 1., 2., tree_i, width, height)
    knots = tree_knots_unit_matrix(rows, columns, year_count, 10, knot_points, 0.3, tree_i, width, height, row)
    list_xy_i = lattice+knots

    # list_x = [i.x for i in list_xy_i]
    # list_y = [i.y for i in list_xy_i]
    # plt.scatter(list_x, list_y)
    # plt.show()

    for i in range(len(list_xy_i)):
        list_xy_i[i].x = x_curve(list_xy_i[i].x, base_width)
        list_xy_i[i].x = x_transform(0., width + base_buffer, list_xy_i[i].x, 0., base_width)
        list_xy_i[i].x += tree_shift_x
        list_xy_i[i].y = x_transform(0., height + base_height, list_xy_i[i].y, 0., rows)
        list_xy_i[i].x -= 2.

    list_xy += list_xy_i
    tree_i += 1
    tree_shift_x += width+buffer+base_buffer
    test += 1
    print(test)
    # if test == 50:
    #     break
#endregion

#region output
import csv
with open(os.path.dirname(__file__) + '/forest.csv', 'w',) as csvfile:
    writer = csv.writer(csvfile, lineterminator = '\n')
    writer.writerow(['index', 'item', 'x', 'y', 'path', 'row', 'column', 'entity', 'tree', 'width_tree', 'height_tree', 'width_knot', 'height_knot', 'year'])
    for i in range(len(list_xy)):
        writer.writerow([list_xy[i].index, list_xy[i].item, list_xy[i].x, list_xy[i].y, list_xy[i].path, list_xy[i].row, list_xy[i].column, list_xy[i].entity, list_xy[i].tree, list_xy[i].w_t, list_xy[i].h_t, list_xy[i].w_k, list_xy[i].h_k, list_xy[i].year])
#endregion

print('finished')