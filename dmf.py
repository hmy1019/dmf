import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import logging

marker_size = 10  #lineとmarkerのsizeは共通、初期値10
R = 3
r = 1
m = 1
theta = []
phi = []
def torus_point(theta, phi):
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return (x, y, z)
def get_theta_phi(x, y, z):
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2) - R)
    return theta, phi
def interpolate_arc(p1, p2,num):
    theta1, phi1 = get_theta_phi(p1[0], p1[1], p1[2])
    theta2, phi2 = get_theta_phi(p2[0], p2[1], p2[2])
    vals = np.linspace(0, 1, num)
    arc_points = []
    for t in vals:
        if abs(theta2 - theta1) > np.pi:
            if theta2 > theta1:
                theta2 -= 2 * np.pi
            else:
                theta2 += 2 * np.pi
        if abs(phi2 - phi1) > np.pi:
            if phi2 > phi1:
                phi2 -= 2 * np.pi
            else:
                phi2 += 2 * np.pi
        theta = (1 - t) * theta1 + t * theta2
        phi = (1 - t) * phi1 + t * phi2
        arc_points.append(torus_point(theta, phi))
    return arc_points

def get_colorscales(category):
    return [
        name for name in dir(getattr(pc, category, None)) 
        if isinstance(getattr(getattr(pc, category, None), name), list) and name != '__all__'
    ]
colorscales = get_colorscales("sequential") + get_colorscales("diverging") + get_colorscales("cyclical")

#######
# DMF #
#######

dmf_all = {}
dmf_max = 1
def calcDMF(grid_size):
    dmf_all = {}
    dmf_v = [0] * grid_size * grid_size
    dmf_e = [0] * grid_size * grid_size * 3
    dmf_f = [0] * grid_size * grid_size * 2
    s = ((grid_size-1) >> 1) + 1
    t = (grid_size >> 1) + 1
    # 左端
    for j in range(t):
        dmf_v[j] = j * 2
        dmf_e[j] = j * 2 + 1
    for j in range(1,s):
        dmf_v[grid_size-j] = j * 2
    for j in range(s-1):
        dmf_e[grid_size-j-1] = j * 2 + 1

    # 左下
    # 0,1-cell
    for i in range(s-1):
        for j in range(t):
            x = i + 1
            y = t - j - 1
            dmf_v[x * grid_size + y] = 4 * t + j + 6 * t * i
            x = i
            dmf_e[x * grid_size + y + grid_size * grid_size] = 2 * t + j + 6 * t * i
    # 1,2-cell
    for i in range(s-1):
        for j in range(t-1):
            x = i
            y = t - j - 2
            dmf_f[x * grid_size + y] = 3 * t + j + 6 * t * i
            dmf_e[x * grid_size + y + grid_size * grid_size * 2] = 5 * t + j + 6 * t * i
            dmf_f[x * grid_size + y + grid_size * grid_size] = 6 * t + j + 6 * t * i
            dmf_e[(x+1) * grid_size + y] = 7 * t + j + 6 * t * i
    # 左上
    # 0,1-cell
    for i in range(s-1):
        for j in range(t-s,t-1):
            x = i + 1
            y = s + j
            dmf_v[x * grid_size + y] = 4 * t + j + 6 * t * i
            x = i
            dmf_e[x * grid_size + y + grid_size * grid_size] = 2 * t + j + 6 * t * i
    # 1,2-cell
    for i in range(s-1):
        for j in range(t-s-1,t-1):
            x = i
            y = s + j
            dmf_f[x * grid_size + y] = 3 * t + j + 6 * t * i
            dmf_e[x * grid_size + y + grid_size * grid_size * 2] = 5 * t + j + 6 * t * i
            dmf_f[x * grid_size + y + grid_size * grid_size] = 6 * t + j + 6 * t * i
            dmf_e[(x+1) * grid_size + y] = 7 * t + j + 6 * t * i
    # 右下
    # 0,1-cell
    for i in range(t-1):
        for j in range(t):
            x = grid_size - i - 1
            y = t - j - 1
            dmf_v[x * grid_size + y] =4 * t + j + 6 * t * i
            x = grid_size - i - 1
            dmf_e[x * grid_size + y + grid_size * grid_size] =2 * t + j + 6 * t * i
    # 1,2-cell
    for i in range(t-1):
        for j in range(1,t):
            x = grid_size - i - 1
            y = t - j - 1
            dmf_f[x * grid_size + y] = 6 * t + j + 6 * t * i - 1
            dmf_e[x * grid_size + y + grid_size * grid_size * 2] = 5 * t + j + 6 * t * i - 1
            dmf_f[x * grid_size + y + grid_size * grid_size] = 3 * t + j + 6 * t * i - 1
            dmf_e[x * grid_size + y] = 7 * t + j + 6 * t * i - 1
    # 右上
    # 0,1-cell
    for i in range(t-1):
        for j in range(t-s+1,t):
            x = grid_size - i - 1
            y = s + j - 1
            dmf_v[x * grid_size + y] = 4 * t + j + 6 * t * i - 1
            x = grid_size - i - 1
            dmf_e[x * grid_size + y + grid_size * grid_size] = 2 * t + j + 6 * t * i - 1
    # 1,2-cell
    for i in range(t-1):
        for j in range(t-s-1,t-1):
            x = grid_size - i - 1
            y = s + j
            dmf_f[x * grid_size + y] = 6 * t + j + 6 * t * i
            dmf_e[x * grid_size + y + grid_size * grid_size * 2] = 5 * t + j + 6 * t * i
            dmf_f[x * grid_size + y + grid_size * grid_size] = 3 * t + j + 6 * t * i
            dmf_e[x * grid_size + y] = 7 * t + j + 6 * t * i
    # 真ん中
    cx = s - 1
    cy = t - 1
    u = dmf_e[grid_size*(cx+1)+cy]
    dmf_f[grid_size*cx+cy] = u + 1
    dmf_f[grid_size*cx+cy+grid_size*grid_size] = u + 1 + (1-t+s)*2
    dmf_e[grid_size*cx+cy+grid_size*grid_size*2] = u
    for j in range(cy):
        y = cy - j - 1
        u = dmf_f[grid_size*cx+y+1+grid_size*grid_size]
        dmf_e[grid_size*cx+y+1+grid_size*grid_size] = u + 1
        dmf_f[grid_size*cx+y] = u + 2
        dmf_e[grid_size*cx+y+grid_size*grid_size*2] = u + 3
        dmf_f[grid_size*cx+y+grid_size*grid_size] = u + 4
    for j in range(grid_size-cy-1):
        y = cy + j + 1
        u = dmf_f[grid_size*cx+y-1]
        dmf_e[grid_size*cx+y+grid_size*grid_size] = u + 1
        dmf_f[grid_size*cx+y+grid_size*grid_size] = u + 2
        dmf_e[grid_size*cx+y+grid_size*grid_size*2] = u + 3
        dmf_f[grid_size*cx+y] = u + 4
    dmf_max = dmf_f[cx*grid_size+grid_size*grid_size]
    dmf_e[cx*grid_size+grid_size*grid_size] = dmf_max - 1
    # marge
    for i in range(grid_size*grid_size):
        dmf_all[f"v_{i}"]=dmf_v[i]
    for i in range(grid_size*grid_size*3):
        dmf_all[f"e_{i}"]=dmf_e[i]
    for i in range(grid_size*grid_size*2):
        dmf_all[f"f_{i}"]=dmf_f[i]
    return dmf_all, dmf_max



def nameCell(grid_size):
    vertices = {}
    edges = {}
    faces = {}
    # 0-cell
    for i in range(grid_size):
        for j in range(grid_size):
            vertices[f'v_{i * grid_size + j}'] = (i, j)
        vertices[f"v_{i * grid_size}'"] = (i, j + 1)
    for j in range(1,grid_size):
        vertices[f"v_{j}'"] = (grid_size, j)
    vertices["v_0''"] = (grid_size, 0.0)
    vertices["v_0'''"] = (grid_size, grid_size)
    # 1-cell
    e_count = 0
    # 縦の線
    for i in range(grid_size):
        for j in range(grid_size - 1):
            v0 = i * grid_size + j
            v1 = v0 + 1
            edges[f'e_{e_count}'] = [f'v_{v0}', f'v_{v1}']
            e_count += 1
        v0 = v1
        v1 = i * grid_size
        edges[f'e_{e_count}'] = [f'v_{v0}', f"v_{v1}'"]
        e_count += 1
    edges["e_0'"] = ["v_0''", "v_1'"]
    for j in range(1, grid_size - 1):
        v0 = j
        v1 = j + 1
        edges[f"e_{j}'"] = [f"v_{v0}'", f"v_{v1}'"]
    v0 = grid_size - 1
    edges[f"e_{grid_size - 1}'"] = [f"v_{v0}'", "v_0'''"]
    # 横の線
    for i in range(grid_size - 1):
        for j in range(grid_size):
            v0 = i * grid_size + j
            v1 = v0 + grid_size
            edges[f'e_{e_count}'] = [f'v_{v0}', f'v_{v1}']
            e_count += 1
        v0 = i * grid_size
        v1 = v0 + grid_size
        edges[f"e_{e_count-grid_size}'"] = [f"v_{v0}'", f"v_{v1}'"]
    v0 = (grid_size - 1) * grid_size
    edges[f'e_{e_count}'] = [f'v_{v0}', "v_0''"]
    e_count += 1
    for j in range(1, grid_size):
        v0 = (grid_size - 1) * grid_size + j
        v1 = j
        edges[f'e_{e_count}'] = [f'v_{v0}', f"v_{v1}'"]
        e_count += 1
    v0 = (grid_size - 1) * grid_size
    edges[f"e_{e_count-grid_size}'"] = [f"v_{v0}'", "v_0'''"]
    # 斜めの線
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v0 = i * grid_size + j
            v1 = v0 + grid_size + 1
            edges[f'e_{e_count}'] = [f'v_{v0}', f'v_{v1}']
            e_count += 1
        v0 = (i + 1) * grid_size - 1
        v1 = v0 + 1
        edges[f'e_{e_count}'] = [f'v_{v0}', f"v_{v1}'"]
        e_count += 1
    for j in range(grid_size - 1):
        v0 = (grid_size - 1) * grid_size + j
        v1 = j + 1
        edges[f'e_{e_count}'] = [f'v_{v0}', f"v_{v1}'"]
        e_count += 1
    v0 = grid_size * grid_size - 1
    edges[f'e_{e_count}'] = [f'v_{v0}', f"v_0'''"]
    e_count += 1
    # 2-cell
    f_count = 0
    keys = list(edges.keys())
    # 上三角
    for i in range(grid_size):
        for j in range(grid_size):
            face_name = f'f_{f_count}'
            f_count += 1
            faces[face_name] = [
                keys[i * grid_size + j],
                keys[(grid_size + 1) * grid_size + (i * grid_size + j) + i + 1],
                keys[(grid_size + 1) * grid_size * 2 + (i * grid_size + j)]
            ]
    # 下三角
    for i in range(grid_size):
        for j in range(grid_size):
            face_name = f'f_{f_count}'
            f_count += 1
            faces[face_name] = [
                keys[i * grid_size + j + grid_size],
                keys[(grid_size + 1) * grid_size + (i * grid_size + j) + i],
                keys[(grid_size + 1) * grid_size * 2 + (i * grid_size + j)]
            ]
    return vertices, edges, faces
def calcCoord(grid_size):
    theta = np.linspace(0, 2 * np.pi, grid_size + 1)[:-1]
    phi = np.linspace(0, 2 * np.pi, grid_size + 1)[:-1]
    coordsOfVerticesOnTorus = {}
    for i in range(grid_size):
        for j in range(grid_size):
            coordsOfVerticesOnTorus[f'v_{i * grid_size + j}'] = torus_point(theta[i],phi[j])
        coordsOfVerticesOnTorus[f"v_{i * grid_size}'"] = torus_point(theta[i],phi[0])
    coordsOfVerticesOnTorus["v_0''"] = torus_point(theta[0],phi[0])
    for j in range(1,grid_size):
        coordsOfVerticesOnTorus[f"v_{j}'"] = torus_point(theta[0],phi[j])
    coordsOfVerticesOnTorus["v_0'''"] = torus_point(theta[0],phi[0])
    return coordsOfVerticesOnTorus

app = Dash(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
@app.callback(
    [Output("graph_triangulation", "figure"), Output("graph_torus", "figure"), Output("graph_colorscale", "figure"), Output('graph_colorscale', 'style')],
    [Input("dropdown", "value"), Input("radio-options", "value"), 
     Input("text-input-gridSize", "value"), Input("text-input-smooth", "value"),
     Input("check-options", "value")]
)
def update(scale, radio_value,grid_size,smooth,check_options):
    dmf_all,dmf_max = calcDMF(grid_size)
    def dmf(cell):
        return dmf_all[cell.rstrip("'")]
    def dmf_color(scale,cell):
        return pc.sample_colorscale(scale, dmf(cell)/dmf_max)[0]
    vertices, edges, faces = nameCell(grid_size)
    coordsOfVerticesOnTorus = calcCoord(grid_size)
    fig_triangulation = go.Figure()
    def face2vertices(face):
        v0 = edges[faces[face][0]][0]
        v1 = edges[faces[face][0]][1]
        v2 = edges[faces[face][1]][0]
        if v2 == v1:
            v2 = edges[faces[face][1]][1]
        return (v0, v1, v2)
    if radio_value == "cell":
        font_size = 10
    else:
        font_size = 20
    # 2-cell
    for face, data in faces.items():
        pos0 = vertices[edges[data[0]][0]]
        pos1 = vertices[edges[data[0]][1]]
        pos2 = vertices[edges[data[1]][0]]
        if pos2 == pos1:
            pos2 = vertices[edges[data[1]][1]]
        x_coords = [pos0[0], pos1[0], pos2[0], pos0[0]]
        y_coords = [pos0[1], pos1[1], pos2[1], pos0[1]]
        if 'showColor' in check_options:
            c = dmf_color(scale, face)
        else:
            c = 'white'
        fig_triangulation.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', hoverinfo='skip',
                                               line=dict(width=0), fill='toself', fillcolor=c))
        if radio_value == "cell" or radio_value == "dmf":
            if radio_value == "cell":
                label = face
            else:
                label = dmf(face)
            mid_x = (pos0[0] + pos1[0] + pos2[0]) / 3
            mid_y = (pos0[1] + pos1[1] + pos2[1]) / 3
            fig_triangulation.add_trace(go.Scatter(x=[mid_x], y=[mid_y], mode='text', hoverinfo='skip', 
                                                   text=[label], textposition='middle center', textfont=dict(color="black",size=font_size)))
    # 1-cell
    for edge, data in edges.items():
        pos0 = vertices[data[0]]
        pos1 = vertices[data[1]]
        if 'showColor' in check_options:
            c = dmf_color(scale, edge)
        else:
            c = 'rgba(0,0,0,0.25)'
        fig_triangulation.add_trace(go.Scatter(x=[pos0[0], pos1[0]], y=[pos0[1], pos1[1]], mode='lines', hoverinfo='skip', 
                                               marker=dict(color=c, size=marker_size)))
        if radio_value == "cell" or radio_value == "dmf":
            if radio_value == "cell":
                label = edge
            else:
                label = dmf(edge)
            mid_x = (pos0[0] + pos1[0]) / 2
            mid_y = (pos0[1] + pos1[1]) / 2
            fig_triangulation.add_trace(go.Scatter(x=[mid_x], y=[mid_y], mode='text', hoverinfo='skip',
                                                   text=[label], textposition='middle center', textfont=dict(color="black",size=font_size)))
    # 0-cell
    for vertex, data in vertices.items():
        if 'showColor' in check_options:
            c = dmf_color(scale, vertex)
        else:
            c = 'rgba(0,0,0,0.25)'
        fig_triangulation.add_trace(go.Scatter(x=[data[0]], y=[data[1]], mode='markers', hoverinfo='skip', 
                                               marker=dict(color=c, size=marker_size)))
        if radio_value == "cell" or radio_value == "dmf":
            if radio_value == "cell":
                label = vertex
            else:
                label = dmf(vertex)
            fig_triangulation.add_trace(go.Scatter(x=[data[0]], y=[data[1]], mode='text', hoverinfo='skip', 
                                                   text=[label], textposition='middle center', textfont=dict(color="black",size=font_size)))
    arrows =[]
    # arrow 0-cell to 1-cell
    for edge, data in edges.items():
        v0 = edges[edge][0]
        v1 = edges[edge][1]
        p0 = vertices[v0]
        p1 = vertices[v1]
        p2 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2) 
        d0 = dmf(v0)
        d1 = dmf(v1)
        d2 = dmf(edge)
        if d0 >= d2:
            quiver_fig = ff.create_quiver(
                x=[p0[0]], y=[p0[1]], 
                u=[p2[0] - p0[0]], v=[p2[1] - p0[1]], 
                scale=0.75, line_color='rgba(0,0,0,0.5)', hoverinfo='skip'
            )
            for trace in quiver_fig.data:
                arrows.append(trace)
            
        if d1 >= d2:
            quiver_fig = ff.create_quiver(
                x=[p1[0]], y=[p1[1]], 
                u=[p2[0] - p1[0]], v=[p2[1] - p1[1]], 
                scale=0.75, line_color='rgba(0,0,0,0.5)', hoverinfo='skip'
            )
            for trace in quiver_fig.data:
                arrows.append(trace)
    # arrow 1-cell to 2cell
    for face, data  in faces.items():
        e0 = faces[face][0]
        e1 = faces[face][1]
        e2 = faces[face][2]
        v0 = edges[e0][0]
        v1 = edges[e0][1]
        v2 = edges[e1][0]
        v3 = edges[e1][1]
        v4 = edges[e2][0]
        v5 = edges[e2][1]
        if v2 == v0 or v2 == v1:
            v = v3
        else:
            v = v2
        d0 = dmf(e0)
        d1 = dmf(e1)
        d2 = dmf(e2)
        d3 = dmf(face)
        p1 = ((vertices[v0][0] + vertices[v1][0] + vertices[v][0]) / 3, (vertices[v0][1] + vertices[v1][1]+ vertices[v][1]) / 3) 
        if d0 >= d3:
            p0 = ((vertices[v0][0] + vertices[v1][0]) / 2, (vertices[v0][1] + vertices[v1][1]) / 2)
            quiver_fig = ff.create_quiver(
                x=[p0[0]], y=[p0[1]], 
                u=[p1[0] - p0[0]], v=[p1[1] - p0[1]], 
                scale=0.75, line_color='rgba(0,0,0,0.5)', hoverinfo='skip'
            )
            for trace in quiver_fig.data:
                arrows.append(trace)
        if d1 >= d3:
            p0 = ((vertices[v2][0] + vertices[v3][0]) / 2, (vertices[v2][1] + vertices[v3][1]) / 2)
            quiver_fig = ff.create_quiver(
                x=[p0[0]], y=[p0[1]], 
                u=[p1[0] - p0[0]], v=[p1[1] - p0[1]], 
                scale=0.75, line_color='rgba(0,0,0,0.5)', hoverinfo='skip'
            )
            for trace in quiver_fig.data:
                arrows.append(trace)
        if d2 >= d3:
            p0 = ((vertices[v4][0] + vertices[v5][0]) / 2, (vertices[v4][1] + vertices[v5][1]) / 2)
            quiver_fig = ff.create_quiver(
                x=[p0[0]], y=[p0[1]], 
                u=[p1[0] - p0[0]], v=[p1[1] - p0[1]], 
                scale=0.75, line_color='rgba(0,0,0,0.5)', hoverinfo='skip'
            )
            for trace in quiver_fig.data:
                arrows.append(trace)
    if 'showArrow' in check_options:
        for trace in arrows:
            fig_triangulation.add_trace(trace)
    fig_triangulation.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x"),
        showlegend=False,
        plot_bgcolor='white',
        height=grid_size * 120,
        width=grid_size * 120,
        margin=dict(l=0, r=0, t=0, b=0)
    )


    fig_torus = go.Figure()
    labelsOfCellOnTorus = []
    #2-cell
    facesOfTorus = []
    for face in faces:
        v0, v1, v2=face2vertices(face)
        pos0=coordsOfVerticesOnTorus[v0]
        pos1=coordsOfVerticesOnTorus[v1]
        pos2=coordsOfVerticesOnTorus[v2]
        arc_points01 = interpolate_arc(pos0,pos1,smooth)
        arc_points02 = interpolate_arc(pos0,pos2,smooth)
        epsilon = 10**-10  #この行を消すと動かない
        for i in range(smooth-1):
            arc_points_apex = interpolate_arc(arc_points01[i],arc_points02[i],i+1)
            arc_points_base = interpolate_arc(arc_points01[i+1],arc_points02[i+1],i+2)
            for j in range(i+1):
                p0 = arc_points_apex[j]
                p1 = arc_points_base[j]
                p2 = arc_points_base[j+1]
                x_coords = [p0[0]+epsilon, p1[0], p2[0]]
                y_coords = [p0[1], p1[1], p2[1]]
                z_coords = [p0[2], p1[2], p2[2]]
                if 'showColor' in check_options:
                    c = dmf_color(scale, face)
                else:
                    c = 'rgba(0,0,0,0)'
                facesOfTorus.append(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    color=c,
                    opacity=1,
                    hoverinfo='skip'
                ))            
        for i in range(smooth-2):
            arc_points_apex = interpolate_arc(arc_points01[i+2],arc_points02[i+2],i+3)
            arc_points_base = interpolate_arc(arc_points01[i+1],arc_points02[i+1],i+2)
            for j in range(i+1):
                p0 = arc_points_apex[j+1]
                p1 = arc_points_base[j]
                p2 = arc_points_base[j+1]
                x_coords = [p0[0]+epsilon, p1[0], p2[0]]
                y_coords = [p0[1], p1[1], p2[1]]
                z_coords = [p0[2], p1[2], p2[2]]
                if 'showColor' in check_options:
                    c = dmf_color(scale, face)
                else:
                    c = 'rgba(0,0,0,0)'
                facesOfTorus.append(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    color=c,
                    opacity=1,
                    hoverinfo='skip'
                ))
    #1-cell
    edgesOfTorus = []
    for i in range(grid_size*grid_size*3):
        # 縦の円弧
        edge = f'e_{i}'
        pos0 = coordsOfVerticesOnTorus[edges[edge][0]]
        pos1 = coordsOfVerticesOnTorus[edges[edge][1]]
        arc_points = interpolate_arc(pos0, pos1,smooth)
        x_vals, y_vals, z_vals = zip(*arc_points)
        if 'showColor' in check_options:
            c = dmf_color(scale, edge)
        else:
            c = 'rgba(0,0,0,0.5)'
        edgesOfTorus.append(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color=c, width=5),
            showlegend=False,
            hoverinfo='skip'
        ))
    verticesOfTorus = []
    #0-cell
    for i in range(grid_size):
        for j in range(grid_size):
            vertex = f'v_{i * grid_size + j}'
            coord = coordsOfVerticesOnTorus[vertex]
            if 'showColor' in check_options:
                c = dmf_color(scale, vertex)
            else:
                c = 'rgba(0,0,0,0.5)'
            verticesOfTorus.append(go.Scatter3d(
                x=[coord[0]], y=[coord[1]], z=[coord[2]],
                mode='markers',
                marker=dict(color=c, size=2),
                hoverinfo='skip'
            ))
            labelsOfCellOnTorus.append(go.Scatter3d(
                x=[coord[0]], y=[coord[1]], z=[coord[2]],
                mode='text',
                text=[vertex], textposition='top center', 
                textfont=dict(color="black", size=font_size),
                hoverinfo='skip'
            ))
    if radio_value == "cell":
        fig_torus = go.Figure(facesOfTorus + edgesOfTorus + verticesOfTorus + labelsOfCellOnTorus)
    elif radio_value == "dmf":
        fig_torus = go.Figure(facesOfTorus + edgesOfTorus + verticesOfTorus)
    else:
        fig_torus = go.Figure(facesOfTorus + edgesOfTorus + verticesOfTorus)
        
    fig_torus.update_layout(
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=(r+m)/(R+r+m)),
        xaxis=dict(showticklabels=False,visible=False, showgrid=False, range=[-R-r-m,R+r+m]),
        yaxis=dict(showticklabels=False,visible=False, showgrid=False, range=[-R-r-m,R+r+m]),
        zaxis=dict(showticklabels=False,visible=False, showgrid=False, range=[-r-m,r+m]),
    ),
    showlegend=False,
    plot_bgcolor='white',
    height=grid_size * 120,
    width=grid_size * 120,
    margin=dict(l=0, r=0, t=0, b=0)
    )


    fig_colorscale = go.Figure()
    for i in range((dmf_max+1)<<4):
        pos0 = ((i>>4)-1/2, 0)
        pos1 = ((i>>4)-1/2, 1)
        pos2 = ((i>>4)+1/2, 0)
        x_coords = [pos0[0], pos1[0], pos2[0]]
        y_coords = [pos0[1], pos1[1], pos2[1]]
        fig_colorscale.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', 
                                 line=dict(width=1,color=pc.sample_colorscale(scale, i/((dmf_max+1)<<4))[0]), fill='toself', 
                                 fillcolor=pc.sample_colorscale(scale, i/((dmf_max+1)<<4))[0], hoverinfo='skip'))
        pos0 = ((i>>4)+1/2, 1)
        x_coords = [pos0[0], pos1[0], pos2[0]]
        y_coords = [pos0[1], pos1[1], pos2[1]]
        fig_colorscale.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', 
                                 line=dict(width=1,color=pc.sample_colorscale(scale, (i+1)/((dmf_max+1)<<4))[0]), fill='toself', 
                                 fillcolor=pc.sample_colorscale(scale, (i+1)/((dmf_max+1)<<4))[0], hoverinfo='skip'))
    fig_colorscale.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=grid_size,
            linewidth=1,
            linecolor='white',
        ),
        yaxis=dict(
            showticklabels=False,
            range=[0,1]
        ),
        showlegend=False,
        plot_bgcolor='white',
        height=40,
        width=grid_size * 120,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    if 'showColor' in check_options:
        style = {'display': 'block'} 
    else:
        style = {'display': 'none'} 
    return fig_triangulation, fig_torus, fig_colorscale, style

app.layout = html.Div([
    html.Div([
        html.Div([
            html.P("Grid Size:", style={'margin-left': '10px', 'display': 'inline-block', 'vertical-align': 'middle'}),
                dcc.Input(
                    id='text-input-gridSize',
                    type='number',
                    min=3,
                    max=16,
                    step=1,
                    value=3,
                    style={'height':'30px','width': '30px', 'display': 'inline-block', 'vertical-align': 'middle'}
            ),
            html.P("Smooth:", style={'margin-left': '10px', 'display': 'inline-block', 'vertical-align': 'middle'}),
                dcc.Input(
                    id='text-input-smooth', 
                    type='number',
                    min=2,
                    max=16,
                    step=1,
                    value=2,
                    style={'height':'30px','width': '30px', 'display': 'inline-block', 'vertical-align': 'middle'}
            ),
            html.P("Color Scale:", style={'margin-left': '10px', 'display': 'inline-block', 'vertical-align': 'middle'}),
                dcc.Dropdown(
                    id='dropdown', 
                    options=colorscales,
                    value='YlGn',
                    style={'height':'30px','width': '200px', 'display': 'inline-block', 'vertical-align': 'middle'}
            ),
            html.Div([
                dcc.RadioItems(
                    id='radio-options',
                    options=[
                        {'label': 'Cell Name', 'value': 'cell'},
                        {'label': 'DMF', 'value': 'dmf'},
                        {'label': 'No Label', 'value': 'none'}
                    ],
                    value='cell',
                    inline=True,
                    style={'display': 'inline-block'}
                ),
            ]),
            html.Div([
                dcc.Checklist(
                    id='check-options',
                    options=[
                        {'label': 'Show Color', 'value': 'showColor'},
                        {'label': 'Show Arrow', 'value': 'showArrow'}
                    ],
                    value=[],
                    inline=True,
                    style={'display': 'inline-block'}
                ),
            ]),
        ])
    ], style={'display': 'flex', 'align-items': 'center'}),
    html.Div([
        html.Div([
            html.P('figure : Triangulation of Torus',
                   style={'margin': '0px','textAlign': 'center'}),
            dcc.Graph(id='graph_triangulation')
        ]),
        html.Div([
            html.P('figure : Torus',
                   style={'margin': '0px','textAlign': 'center'}),
            dcc.Graph(id='graph_torus')
        ])
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(
        id='graph_colorscale',
        style={'display': 'block'}
    ),
])
#port 8050を使用できない場合は変更
app.run_server(host='0.0.0.0', port=8050)