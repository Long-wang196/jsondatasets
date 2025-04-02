import concurrent.futures
import json
import math, time, ujson
import os,random, fiona
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from shapely import LineString
from shapely.geometry import shape, Point, Polygon, MultiPolygon, mapping
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, ConvexHull,QhullError
from scipy.optimize import minimize
from shapely.ops import unary_union, snap
from joblib import Parallel, delayed
from rtree import index

def read_shapfile(shapefile_path):
    #存储几何形状的列表
    geometries = []
    #存储坐标的字典
    coordinates = defaultdict(list)

    with fiona.open(shapefile_path, 'r') as shapefile:
        for feature in shapefile:
            geom = shape(feature['geometry'])
            geometries.append(geom)

            #提取坐标数据并存储
            if isinstance(geom, Polygon):
                coordinates['Polygon'].append(list(geom.exterior.coords))
            elif isinstance(geom, MultiPolygon):
                for poly in geom:
                    coordinates['MultiPolygon'].append(list(poly.exterior.coords))
            elif isinstance(geom, Point):
                coordinates['Point'].append((geom.x, geom.y))
            elif isinstance(geom, LineString):
                coordinates['LineString'].append(list(geom.coords))
    return geometries, coordinates

'''Centroids Calcualtion'''

def calculate_centroid(polygon):
    return polygon.centroid

def calculate_centroids(multi_polygon):
    # 创建 Polygon 对象
    polygons = [Polygon(polygon) for polygon in multi_polygon]

    # 并行计算质心
    centroids = Parallel(n_jobs=-1)(delayed(calculate_centroid)(polygon) for polygon in polygons)

    # 提取质心坐标
    centroids_coords = [(centroid.x, centroid.y) for centroid in centroids]

    return centroids_coords

'''calculation Shared edge'''
def has_shared_delaunay_edge(delaunay, polygons, centroids):
    shared_edges = []
    polygon_neighbors = {i: set() for i in range(len(polygons))}

    # 提前将 polygons 转换为 Polygon 对象
    polygons = [Polygon(poly) for poly in polygons]
    centroids = np.array(centroids)  # 使用 NumPy 数组加速索引

    # 创建R树索引以加速质心查找
    idx = index.Index()
    for i, centroid in enumerate(centroids):
        idx.insert(i, (centroid[0], centroid[1], centroid[0], centroid[1]))

    for simplex in delaunay.simplices:
        indices = [simplex[0], simplex[1], simplex[2]]
        pairs = [(indices[0], indices[1]), (indices[1], indices[2]), (indices[2], indices[0])]

        for i, j in pairs:
            if i >= len(centroids) or j >= len(centroids):
                continue

            point_i = Point(centroids[i])
            point_j = Point(centroids[j])

            # 使用R树索引查找多边形
            possible_polys_i = list(idx.intersection((point_i.x, point_i.y, point_i.x, point_i.y)))
            possible_polys_j = list(idx.intersection((point_j.x, point_j.y, point_j.x, point_j.y)))

            poly1 = poly2 = None
            for k in possible_polys_i:
                if point_i.within(polygons[k]):
                    poly1 = k
                    break

            for k in possible_polys_j:
                if point_j.within(polygons[k]):
                    poly2 = k
                    break

            if poly1 is not None and poly2 is not None and poly1 != poly2:
                shared_edges.append(((centroids[i], centroids[j]), (poly1, poly2)))
                polygon_neighbors[poly1].add(poly2)
                polygon_neighbors[poly2].add(poly1)

    return shared_edges, polygon_neighbors

def minimum_bounding_rectangle(polygons):
    """
    计算多个多边形的最小面积外接矩形

    参数：
    polygons (list): 多个多边形的顶点坐标列表，如 [[(x1, y1), (x2, y2), ...], ...]

    返回：
    各种与最小面积外接矩形相关的属性字典
    """
    min_rect_coords = {}
    width = {}
    height = {}
    area = {}
    area_poly = {}
    perimeter_poly = {}
    perimeter_min_rect = {}
    longaxis = {}

    for i, poly_coords in enumerate(polygons):
        poly = Polygon(poly_coords)
        min_rect = poly.minimum_rotated_rectangle
        min_x, min_y, max_x, max_y = min_rect.bounds

        min_rect_coords[i] = Polygon(list(min_rect.exterior.coords))
        width[i] = max_x - min_x
        height[i] = max_y - min_y
        longaxis[i] = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        area[i] = min_rect.area
        area_poly[i] = poly.area
        perimeter_poly[i] = poly.length
        perimeter_min_rect[i] = min_rect.length

    return (min_rect_coords, width, height, area, area_poly,
            perimeter_poly, longaxis, perimeter_min_rect)
def compute_minimum_distances(polygons, adjacency_dict):
    """
    计算每个多边形与其他多边形之间的最小距离。

    参数:
    polygons (dict): 以多边形ID为键，Polygon对象为值的字典。
    adjacency_dict (dict): 以多边形ID为键，邻接多边形ID集合为值的字典。

    返回:
    np.ndarray: 最小距离矩阵。
    """
    polygon_ids = list(polygons.keys())
    n = len(polygon_ids)
    distance_matrix = np.full((n, n), np.inf)

    for i, j in combinations(range(n), 2):
        poly_id = polygon_ids[i]
        neighbor_id = polygon_ids[j]
        if neighbor_id in adjacency_dict[poly_id]:
            distance = polygons[poly_id].distance(polygons[neighbor_id])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix, polygon_ids

def compare_distance_and_bounding_box(polygons, adjacency_dict):
    """
    比较每个多边形与其他多边形的最小距离与其最小面积外接矩形的最短边的大小。

    参数:
    polygons (dict): 以多边形ID为键，Polygon对象为值的字典。
    adjacency_dict (dict): 以多边形ID为键，邻接多边形ID集合为值的字典。

    返回:
    list: 包含比较结果的列表，每个元素是一个元组 (poly_id, neighbor_id, distance, min_side)。
    """
    distance_matrix, polygon_ids = compute_minimum_distances(polygons, adjacency_dict)
    id_min_rect_coords, width, height, area, area_poly, id_min_rec_perimeter_poly, id_min_rec_longaxis, perimeter_min_rect = minimum_bounding_rectangle([polygons[id].exterior.coords for id in polygon_ids])

    comparisons = []
    Clo_nei_pair = []
    rel_clo_nei_pair = []
    non_clo_nei_pair = []

    for i, poly_id in enumerate(polygon_ids):
        min_side = min(width[i], height[i])
        polygon = polygons[poly_id]

        for j, neighbor_id in enumerate(polygon_ids):
            if poly_id != neighbor_id and distance_matrix[i, j] != np.inf:
                distance = distance_matrix[i, j]
                comparisons.append((poly_id, neighbor_id, distance, min_side))
                if distance <= min_side:
                    Clo_nei_pair.append((poly_id, list(polygon.exterior.coords)))
                elif min_side < distance <= 2 * min_side:
                    rel_clo_nei_pair.append((poly_id, list(polygon.exterior.coords)))
                else:
                    non_clo_nei_pair.append((poly_id, list(polygon.exterior.coords)))

    return distance_matrix, comparisons, Clo_nei_pair, rel_clo_nei_pair, non_clo_nei_pair

# 计算多边形内插点的函数
def generate_edge_points(coords, k):
    n = len(coords)
    for i in range(n - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        yield p1
        dx = (p2[0] - p1[0]) / (k + 1)
        dy = (p2[1] - p1[1]) / (k + 1)
        for j in range(1, k + 1):
            yield (p1[0] + j * dx, p1[1] + j * dy)
    yield coords[-1]  # 最后一个点

def bounded_voronoi(points):
    # 创建矩形边界
    min_x, min_y = np.min(points, axis=0) - 10
    max_x, max_y = np.max(points, axis=0) + 10
    boundary = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    boundary_points = np.array(boundary.exterior.coords)
    all_points = np.vstack([points, boundary_points])

    # 生成Voronoi图
    vor = Voronoi(all_points)

    # 存储每个点的Voronoi单元
    point_to_polygon = {}
    for i, region_index in enumerate(vor.point_region):
        if region_index <= -1:
            continue
        vertices = [vor.vertices[v]
                    for v in vor.regions[region_index]
                    if v != -1 and v < len(vor.vertices)]
        if len(vertices) >= 3:
            polygon = Polygon(vertices)
            # 仅保留在边界内的部分
            clipped_polygon = polygon.intersection(boundary)
            if not clipped_polygon.is_empty and i < len(points):
                point_to_polygon[i] = clipped_polygon

    return point_to_polygon

def merged_voronoi(interpolated_points):
    # 提取点集和id
    all_points = []
    id_dict_points = {}
    id_to_point_indices = {}

    point_index = 0
    for id, points in interpolated_points:
        # 过滤掉无效的点，并确保点的格式正确
        valid_points = [(point[0], point[1]) for point in points if
                        isinstance(point, (list, tuple)) and len(point) == 2]

        id_dict_points[id] = valid_points
        point_indices = list(range(point_index, point_index + len(valid_points)))

        all_points.extend(valid_points)  # 使用 extend 而不是 append 可以减少循环

        id_to_point_indices[id] = point_indices
        point_index += len(valid_points)

    all_points = np.array(all_points)

    all_points = np.array(all_points)
    # 生成受限于边界的Voronoi图
    point_to_polygon = bounded_voronoi(all_points)

    id_polygon_with = {id: [point_to_polygon[i]
                            for i in indices
                            if i in point_to_polygon]
                       for id, indices in id_to_point_indices.items()
                       }

    merged_polygons = {}
    merged_polygons_area_Voronoi = {}
    for id_poly, polys in id_polygon_with.items():
        if len(polys) > 1:
            merged_polygon = unary_union(polys).buffer(0)  # 使用 buffer(0) 修复拓扑问题
        else:
            merged_polygon = polys[0]
        merged_polygons[id_poly] = merged_polygon
        merged_polygons_area_Voronoi[id_poly] = merged_polygon.area

    # center = vor.points.mean(axis=0)
    # 绘制 Voronoi 图
    # fig, ax = plt.subplots()
    # # 绘制 Voronoi 图的边界和种子点
    # voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    #
    # # 绘制种子点
    # ax.plot(all_points[:, 0], all_points[:, 1], 'ro')  # 'ro'表示红色圆点

    # 设置图形显示范围
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])

    return merged_polygons, merged_polygons_area_Voronoi

# 设置随机颜色函数
def random_color():
    return (random.random(), random.random(), random.random())


def convexity_calculation(poi):
    id_convex_hull_area = {}
    id_convex_hull_perimeter = {}
    for i, points in enumerate(poi):
        polygon = Polygon(points)
        convex_hull = polygon.convex_hull
        id_convex_hull_area[i] = convex_hull.area
        id_convex_hull_perimeter[i] = convex_hull.length
    return id_convex_hull_area, id_convex_hull_perimeter

# 函数：计算多边形的最大内切圆
def largest_incircle(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    max_radius = 0
    best_center = None
    for x in np.linspace(minx, maxx, 100):
        for y in np.linspace(miny, maxy, 100):
            point = Point(x, y)
            if polygon.contains(point):
                radius = polygon.exterior.distance(point)
                if radius > max_radius:
                    max_radius = radius
                    best_center = point
    return best_center, max_radius

# 函数circumcircle和smallest_enclosing_circle：计算多边形的外接圆
def circumcircle(points):
    def objective(center):
        return max(np.linalg.norm(points - center, axis=1))

    center_guess = np.mean(points, axis=0)
    result = minimize(objective, center_guess, method='Nelder-Mead')
    center = result.x
    radius = objective(center)
    return center, radius

def smallest_enclosing_circle(polygon):
    points = np.array(polygon.exterior.coords)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    center, radius = circumcircle(hull_points)
    return center, radius

def calculate_factors(id_d, voronoi_area, id_min_rect_area_poly, id_min_rect_area,
                      id_min_rect_height, id_min_rect_width, id_min_rec_perimeter_poly,
                      id_convex_hull_perimeter, id_convex_hull_area, id_min_rec_longaxis, poi):
    value = voronoi_area[id_d]
    min_rect_area_poly = id_min_rect_area_poly[id_d]
    min_rect_area = id_min_rect_area[id_d]
    min_rect_height = id_min_rect_height[id_d]
    min_rect_width = id_min_rect_width[id_d]
    min_rec_perimeter_poly = id_min_rec_perimeter_poly[id_d]
    convex_hull_perimeter = id_convex_hull_perimeter[id_d]
    convex_hull_area = id_convex_hull_area[id_d]
    min_rec_longaxis = id_min_rec_longaxis[id_d]
    polygon = Polygon(poi[id_d])

    d = (value - min_rect_area_poly) / value
    R = min_rect_area_poly / min_rect_area
    Ar = min_rect_height / min_rect_width
    Pc = min_rec_perimeter_poly / convex_hull_perimeter
    Ac = min_rect_area_poly / convex_hull_area
    Sp = (4 * math.pi * min_rect_area_poly) / (convex_hull_perimeter ** 2)
    Ec = min_rec_longaxis / min_rect_width
    Sf = (4 * math.pi * min_rect_area_poly) / (min_rec_perimeter_poly ** 2)
    incircle_center, incircle_radius = largest_incircle(polygon)
    outcircle_center, outcircle_radius = smallest_enclosing_circle(polygon)
    C = incircle_radius / outcircle_radius

    return id_d, d, R, Ar, Pc, Ac, Sp, Ec, C, Sf

def plot_polygon(polygon, ax, edge_color):
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=1, fc='none', ec=edge_color)
    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms: # 使用 geoms 迭代 MultiPolygon
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=1, fc='none', ec=edge_color)
    else:
        print(f"Unsupported geometry type: {type(polygon)}")

# 假设 json_data 是一个包含复杂数据结构的列表
def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, (Polygon, MultiPolygon)):
        return mapping(data)  # 将 shapely 对象转换为 GeoJSON 格式
    return data

def write_json_file(json_data, file_path):
    serializable_data = convert_to_serializable(json_data)

    try:
        with open(file_path, 'w') as f:
            ujson.dump(serializable_data, f, indent=4)
        print(f'存储数据为 {file_path} 完成')
    except (IOError, OSError) as e:
        print(f"File operation error: {e}")
    except (TypeError, ValueError) as e:
        print(f"Data serialization error: {e}")

def merge_polygons_rtree(polygons, tolerance=1e-8):
    if not polygons:
        return []

    # 创建 rtree 索引
    idx = index.Index()

    for pos, poly in enumerate(polygons):
        idx.insert(pos, poly.bounds)

    snapped_polygons = []

    # 使用 rtree 查找相邻多边形并处理共边情况
    for i, poly in enumerate(polygons):
        neighbors = [j for j in idx.intersection(poly.bounds) if i != j]
        for j in neighbors:
            other_poly = polygons[j]
            snapped_polygons.append(snap(poly, other_poly, tolerance))
        snapped_polygons.append(poly)

    # 使用 unary_union 来合并多边形
    merged = unary_union(snapped_polygons)

    # 确保结果是一个 MultiPolygon
    if isinstance(merged, Polygon):
        merged = MultiPolygon([merged])

    return list(merged.geoms)

def process_polygon(i, polygon, flattened_distance_matrix, Clo_nei_set, rel_clo_nei_set):
    coords = np.array(polygon.exterior.coords)
    if i in Clo_nei_set:
        lambda_value = flattened_distance_matrix[i]
        k = max(1,int(polygon.length / (50 * lambda_value)))
    elif i in rel_clo_nei_set:
        lambda_value = flattened_distance_matrix[i] * 2
        k = max(1,int(polygon.length / (100 * lambda_value)))
    else:
        return i, coords.tolist()

    points = generate_edge_points(coords, k)  # 返回生成器
    # points = list(points_generator)  # 将生成器转为列表

    return i, points

def parallel_interpolation(polygons, flattened_distance_matrix, Clo_nei_pair, rel_clo_nei_pair):
    Clo_nei_set = set(x for x, _ in Clo_nei_pair)
    rel_clo_nei_set = set(x for x, _ in rel_clo_nei_pair)

    with ThreadPoolExecutor() as executor:  # 使用 ThreadPoolExecutor
        futures = [executor.submit(process_polygon, i, polygon, flattened_distance_matrix, Clo_nei_set, rel_clo_nei_set)
                   for i, polygon in enumerate(polygons)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results

def visibility_factors_calculation(readshpfile, factors_json_data):

    '''
    :param readshpfile: 输入的shapefile数据并读取其坐标
    :return: 将其写为json文件，并打印写入成功
    '''

    geo, shpcoor = read_shapfile(readshpfile)

    start_time = time.time()
    print(f'running in merge_polygons')
    merged_polygons = merge_polygons_rtree(geo)
    end_time = time.time()
    print(f'The time of running in merge_polygons is: {"%.2f" % (end_time - start_time)} secnods')
    print('End of running in merge_polygons\n', '-' * 100)

    start_time = time.time()
    print(f'running in merged_coords')
    poi = [
        list(zip(polygon.exterior.xy[0], polygon.exterior.xy[1]))
        for polygon in merged_polygons
        if isinstance(polygon, Polygon)
    ]
    end_time = time.time()
    print(f'The time of running in merged_coords is: {"%.2f" % (end_time - start_time)} secnods')
    print('End of running in merged_coords\n', '-' * 100)

    # 计算质心
    print('runing in def calculate_centroids')
    centroids = calculate_centroids(poi)
    print('the end of runing in def calculate_centroids\n', '-' * 100)

    # Compute Delaunay triangulation of centroids
    points = np.array(centroids)
    delaunay = Delaunay(points)
    # Check for shared edges
    print('runing in def has_shared_delaunay_edge')
    startTime = time.time()
    shared_edges, polygon_neighbors = has_shared_delaunay_edge(delaunay, poi, centroids)
    endTime = time.time()
    # print(polygon_neighbors)
    print('time for running def has_shared_delaunay_edge is:', endTime - startTime)
    print('the end of runing in def has_shared_delaunay_edge\n', '-' * 100)

    # 计算并绘制多个多边形的面积最小外接矩形
    print('runing in def minimum_bounding_rectangle')
    (id_min_rect_coords, id_min_rect_width,id_min_rect_height,
     id_min_rect_area, id_min_rect_area_poly, id_min_rec_perimeter_poly,
     id_min_rec_longaxis, perimeter_min_rect) \
        = minimum_bounding_rectangle(poi)
    print('the end of runing in def minimum_bounding_rectangle\n', '-' * 100)

    # 调用函数进行分类
    print('runing in def compare_distance_and_bounding_box')
    distance_matrix, comparisons, Clo_nei_pair, rel_clo_nei_pair, non_clo_nei_pair \
        = compare_distance_and_bounding_box(id_min_rect_coords, polygon_neighbors)
    print('the end of runing in def compare_distance_and_bounding_box\n', '-' * 100)

    # 结果存储
    print('runing in interpolated_points')
    start_time = time.time()
    # print(Clo_nei_pair)
    polygons = [Polygon(i) for i in poi]

    # 预计算 distance_matrix
    # 预计算 distance_matrix
    flattened_distance_matrix = np.array([np.min(dm[dm != 0]) for dm in distance_matrix])
    interpolated_points = parallel_interpolation(polygons, flattened_distance_matrix, Clo_nei_pair, rel_clo_nei_pair)

    endtime = time.time()
    print('The time of runing in interpolated_points is:', endtime - start_time)
    print('the end of runing in interpolated_points\n', '-' * 100)

    print('runing in def merged_voronoi')
    startTime2 = time.time()
    voronoi_polygons, voronoi_area = merged_voronoi(interpolated_points)
    endTime2 = time.time()
    print('The time of runing in def merged_voronoi is:', endTime2 - startTime2)
    print('the end of runing in def merged_voronoi\n', '-' * 100)

    # 绘制多边形
    # 可视化结果
    # fig, ax = plt.subplots()
    # for polygon in voronoi_polygons.values():
    #     fill_color = random_color()  # 每个多边形随机填充颜色
    #     edge_color = 'b'  # 设置边界线颜色为蓝色
    #     try:
    #         plot_polygon(polygon, ax, edge_color)
    #     except Exception as e:
    #         print(f"Error plotting polygon: {e}")
    # # 绘制点并用ID标注
    # for pois in poi:
    #     x, y = zip(*pois)
    #     # print((x,y))
    #     edge_color = 'r'  # 设置边界线颜色为蓝色
    #     ax.fill(x, y, alpha=0.5, fc='none', ec=edge_color)
    #         # ax1.text(x + 0.2, y, point_id, fontsize=12)
    #
    # plt.tight_layout()
    # plt.show()

    # print(voronoi_polygons)
    print('runing in def convexity_calculation')
    id_convex_hull_area, id_convex_hull_perimeter = convexity_calculation(poi)
    print('the end of runing in def convexity_calculation\n', '-' * 100)

    # 计算邻视性因子、几何特征和因子尺寸
    print('running in all factors')
    startTime2 = time.time()

    results = Parallel(n_jobs=-1)(
        delayed(calculate_factors)(id_d, voronoi_area, id_min_rect_area_poly, id_min_rect_area, id_min_rect_height,
                                   id_min_rect_width, id_min_rec_perimeter_poly, id_convex_hull_perimeter,
                                   id_convex_hull_area, id_min_rec_longaxis, poi) for id_d in voronoi_area.keys())

    # 处理结果
    id_Denisty, id_rectangualrity, id_Aspect_ratio, id_perimeter_convexity, id_area_convexity = {}, {}, {}, {}, {}
    id_sphericity, id_ecentricity, id_circularity, id_shape_factor = {}, {}, {}, {}

    for result in results:
        id_d, d, R, Ar, Pc, Ac, Sp, Ec, C, Sf = result
        id_Denisty[id_d] = d
        id_rectangualrity[id_d] = R
        id_Aspect_ratio[id_d] = Ar
        id_perimeter_convexity[id_d] = Pc
        id_area_convexity[id_d] = Ac
        id_sphericity[id_d] = Sp
        id_ecentricity[id_d] = Ec
        id_circularity[id_d] = C
        id_shape_factor[id_d] = Sf

    endTime2 = time.time()
    print(f'The time of running all factors is: {"%.2f" % (endTime2 - startTime2)} secnods' )
    print('the end of running all factors\n', '-' * 100)

    # 计算每个多边形的唯一id值
    ids_polygons = [f"id{i}" for i in range(len(poi))]

    # 三大因子名称
    id_visibility = {
        'id_Denisty': id_Denisty, 'id_area_origin': id_min_rect_area_poly, 'id_area_voronoi': voronoi_area
    }

    id_geometry = {
        'id_rectangualroty': id_rectangualrity, 'id_Aspect_ratio': id_Aspect_ratio,
        'id_perimeter_convexity': id_perimeter_convexity, 'id_area_convexity': id_area_convexity,
        'id_sphericity': id_sphericity, 'id_ecentricity': id_ecentricity,
        'id_circularity': id_circularity, 'id_shape_factor': id_shape_factor
    }

    id_size = {
        'id_perimeter_oripoly': id_min_rec_perimeter_poly,
        'id_min_rect_area': id_min_rect_area,
        'id_mini_rect_perimeter': perimeter_min_rect
    }

    # 生成数据结构
    json_data = {}
    for i, id_ in enumerate(ids_polygons):
        # i = int(id_[2:])
        json_data[id_] = {
            'ID': i,
            'coordinates': poi[i],
            'Siginificance': None,
            'Visibility': None,
            'Visibility_details': {factor: id_visibility[factor][i] for factor in id_visibility.keys()},
            'Geometry': None,
            'Geometry_details': {factor: id_geometry[factor][i] for factor in id_geometry.keys()},
            'Size': None,
            'Size_details': {factor: id_size[factor][i] for factor in id_size.keys()}
        }
    # print(json_data)

    # serializable_data = convert_to_serializable(json_data)
    # 检查转换后的数据
    # print("Serializable Data Sample:",
    #       serializable_data[:2]
    #       if isinstance(serializable_data, list)
    #       else list(serializable_data.items())[:2])
    # print('-' * 100)

    return write_json_file(json_data, factors_json_data)

def ReaddataFromJSON(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def weightCalculation(data):

    P = data / data.sum(axis=0) # P表示不同因子在全部因子中所占的比重
    P = np.clip(P, a_min=1e-10, a_max=None) # 避免对0取对数的问题，将P中小于1e-10的值替换为1e-10

    try:
        E = -np.nansum(P * np.log(P), axis=0) / np.log(len(data)) # E表示熵值
    except Exception as e:
        print(f"熵值计算错误: {e}")
        E = np.zeros(data.shape[1])

    # 处理 NaN 和 -inf 值（当 P 中某些值为0时会出现）
    E = np.nan_to_num(E)

    # 权重计算
    D = 1 - E # D为权重的分子表达式
    weights = D / D.sum()

    return E, weights

def max_min_nomarilzation(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def fusion_factors(read_files, write_files1, write_files2):

    data = ReaddataFromJSON(read_files)
    id_visibility = []
    id_geometry = []
    id_size = []

    for mainValue in data.values():
        # print(mainValue['id_visibility'])
        id_visibility.append(list(mainValue['Visibility_details'].values()))
        id_geometry.append(list(mainValue['Geometry_details'].values()))
        id_size.append(list(mainValue['Size_details'].values()))

    visibility_arr = np.array(max_min_nomarilzation(id_visibility))
    geometry_arr = np.array(max_min_nomarilzation(id_geometry))
    size_arr = np.array(max_min_nomarilzation(id_size))

    # 不同因子特征的权重值，即V、G和S子特征的权重，数量分别为3、8和3
    E_visibility, w_visibility_factors = weightCalculation(visibility_arr)
    E_geometry, w_gemetry_factors = weightCalculation(geometry_arr)
    E_size, w_size_factors = weightCalculation(size_arr)

    # print(f'The weights of the id_visibility is {visibility_factors}, respectively')
    # print(f'The weights of the id_visibility is {gemetry_factors}, respectively')
    # print(f'The weights of the id_visibility is {size_factors}, respectively')
    # print(size_arr)

    # 可视性、几何特征和尺寸因子的值
    V = np.sum(w_visibility_factors.T * visibility_arr, axis=1)
    G = np.sum(w_gemetry_factors.T * geometry_arr, axis=1)
    S = np.sum(w_size_factors.T * size_arr, axis=1)

    # 计算可视性、几何特征和尺寸因子的权重值，即w_VGS为三者的权重矩阵，第一列为可视性权重，依此类推
    VGS = np.column_stack([V, G, S])
    E_VGS, w_VGS = weightCalculation(VGS)

    # 可视性增强器
    Siginicicance = np.sum(w_VGS * VGS, axis=1)
    # print(Siginicicance)

    for key in data:
        index = int(key[2:])
        item = data[key]
        if index < len(Siginicicance):
            item['Siginificance'] = Siginicicance[index]

        if 'Visibility' in item and index < len(V):
            item['Visibility'] = V[index]
        if 'Geometry' in item and index < len(G):
            item['Geometry'] = G[index]
        if 'Size' in item and index < len(S):
            item['Size'] = S[index]

    visibility = {
        'id_Denisty': {}, 'id_area_origin': {}, 'id_area_voronoi': {}
    }

    geometry = {
        'id_rectangualroty': {}, 'id_Aspect_ratio': {},
        'id_perimeter_convexity': {}, 'id_area_convexity': {},
        'id_sphericity': {}, 'id_ecentricity': {},
        'id_circularity': {}, 'id_shape_factor': {}
    }

    size = {
        'id_perimeter_oripoly': {},
        'id_min_rect_area': {},
        'id_mini_rect_perimeter': {}
    }

    json_data = {'Visibility':{}, 'Geometry':{},'Size':{}}
    for id_json, values_json in enumerate(json_data.values()):
        values_json['Weights'] = round(w_VGS[id_json], 4)
        values_json['Entropy'] = round(E_VGS[id_json], 4)
        values_json['EffectiveValue'] = round(1 - E_VGS[id_json], 4)

    for i, key in enumerate(visibility):
        json_data['Visibility'][key] = {
            'Weights': round(w_visibility_factors[i], 2),
            'Entropy': round(E_visibility[i], 4),
            'EffectiveValue': round(1 - E_visibility[i], 4)
        }

    for i, key in enumerate(geometry):
        json_data['Geometry'][key] = {
            'Weights': round(w_gemetry_factors[i], 4),
            'Entropy': round(E_geometry[i], 4),
            'EffectiveValue': round(1 - E_geometry[i], 4)
        }

    for i, key in enumerate(size):
        json_data['Size'][key] = {
            'Weights': round(w_size_factors[i], 4),
            'Entropy': round(E_size[i], 4),
            'EffectiveValue': round(1 - E_size[i], 4)
        }

    return  write_json_file(data, write_files1), write_json_file(json_data, write_files2)


def main():
    os.chdir('D:\Python environment\TestDataShapfile\ExperimentData')
    ori_shp = 'Export_Output_2.shp'
    write_data_as_json = 'Including_salienceAttribite.json'
    write_filetes_weightsetal = 'Weights_Entropy_Effectivevalue.json'
    visibility_factors_calculation(ori_shp, write_data_as_json)
    fusion_factors(write_data_as_json, write_data_as_json, write_filetes_weightsetal)

    # folder_path = 'D:\Python environment\TestDataShapfile\ExperimentData'
    # # 获取所有 .shp 文件和 .json 文件
    # shp_files = [f for f in os.listdir(folder_path) if f.endswith('.shp')]
    #
    # # 循环读取 .shp 文件并处理
    # for shp_file in shp_files:
    #     # 构造文件名
    #     base_name = os.path.splitext(shp_file)[0]
    #     write_data_as_json = f'{base_name}_Fusion.json'
    #     write_filetes_weightsetal = f'{base_name}_Weights_Entropy_Effectivevalue.json'
    #
    #     # 构造完整路径
    #     shp_file_path = os.path.join(folder_path, shp_file)
    #     json_file_path = os.path.join(folder_path, write_data_as_json)
    #     weight_file_path = os.path.join(folder_path, write_filetes_weightsetal)
    #
    #     # 调用函数处理文件
    #     visibility_factors_calculation(shp_file_path, json_file_path)
    #     fusion_factors(json_file_path, json_file_path, weight_file_path)


if __name__ == "__main__":
    main()
