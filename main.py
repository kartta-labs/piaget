# Lint as: python3
"""TODO(sasantv): DO NOT SUBMIT without one-line documentation for main.

TODO(sasantv): DO NOT SUBMIT without a detailed description of main.
"""
import csv
from operator import itemgetter
import networkx as nx
import numpy as np
from scipy.stats import multivariate_normal


class Point(object):
  def __init__(self, x: float, y: float):
    '''Initializes point with x and y values'''
    self.x = x
    self.y = y

class Rectangle(object):
  def __init__(self, bottom_left: Point, top_right: Point):
    '''Initializes point with x and y values'''
    self.bottom_left = bottom_left
    self.top_right = top_right


class Grid(object):
  def __init__(self, bbox: Rectangle, ny, nx):
    '''Creates a grid given a bbox and number of cells in each direction'''
    self.bbox = bbox
    self.nx = nx
    self.ny = ny
    self.dx = float(bbox.top_right.x - bbox.bottom_left.x)/float(nx)
    self.dy = float(bbox.top_right.y - bbox.bottom_left.y)/float(ny)
    self.construct_grid()
  
  def construct_grid(self):
    x_offset = self.dx/2
    y_offset =  self.dy/2

    self.x, self.y = np.mgrid[self.bbox.bottom_left.x + x_offset:self.bbox.top_right.x:self.dx, self.bbox.bottom_left.y+y_offset:self.bbox.top_right.y:self.dy]
    self.pos = np.dstack((self.x, self.y))
    self.cells = np.ones((self.ny, self.nx), dtype=float)

  def get_cell(x, y):
    row = int((y - self.bbox.bottom_left.y)/dy)
    col = int((x - self.bbox.bottom_left.x)/dx)
    return self.cells[row][col]

  def set_cell(x, y, value):
    row_index = int((y - self.bbox.bottom_left.y)/dy)
    col_index = int((x - self.bbox.bottom_left.x)/dx)
    self.cells[row_index][col_index] = value

class ProbabilityGrid(object):
  def __init__(self, bbox: Rectangle, ny, nx, normalize: bool = True):
    self.grid = Grid(bbox, ny, nx)
    if normalize:
      self.normalize()

  def overwrite_with_normal_dist(self, mean, cov, normalize: bool = True):
    rv = multivariate_normal(mean=mean, cov=cov)
    self.grid.cells = rv.pdf(self.grid.pos)
    if normalize:
      self.normalize
  def probability_sum_check(self):
    return np.sum(self.grid.cells) * (self.grid.dx * self.grid.dy)

  def normalize(self):
    norm = self.probability_sum_check()
    if norm == 0:
        norm = np.finfo(self.grid.cells.dtype).eps
    self.grid.cells /= norm
  
  def posterior(self, likelihood, normalize: bool = True):
    self.grid.cells *= likelihood
    if normalize:
      self.normalize()


def create_attribute_dictionaries(nodes_header, nodes):
  dict_of_attr = {}
  for i in range(1, len(nodes_header)):
    attributes = {}
    for node in nodes:
      attributes[node[0]] = float(node[i])
    dict_of_attr[nodes_header[i]] = attributes
  return dict_of_attr

def bbox_intersection(min_x_0, min_y_0, max_x_0, max_y_0, min_x_1, min_y_1, max_x_1, max_y_1):
  min_x = max(min_x_0, min_x_1)
  min_y = max(min_y_0, min_y_1)
  max_x = min(max_x_0, max_x_1)
  max_y = min(max_y_0, max_y_1)
  area = abs(max((max_x - min_x, 0)) * max((max_y - min_y), 0))
  if area == 0:
    return [None, None, None, None]
  return [min_x, min_y, max_x, max_y]

def main():
  bottom_left = Point(-1,-1)
  top_right = Point(1,1)
  nx = 10
  ny = 10
  bbox = Rectangle(bottom_left, top_right)
  prob_grid = ProbabilityGrid(bbox, nx, ny)
  prob_grid.overwrite_with_normal_dist(mean=[0,0], cov=[[1, 0], [0, 1]], normalize=False)
  print(prob_grid.probability_sum_check())
  prob_grid.normalize()
  print(prob_grid.probability_sum_check())
  shape = prob_grid.grid.cells.shape
  prob_grid.posterior(prob_grid.grid.cells * np.random.rand(shape[0],shape[1]), normalize=False)

  print(prob_grid.probability_sum_check())
  prob_grid.normalize()
  print(prob_grid.probability_sum_check())

  return
  # Read the nodes csv file
  with open('data/nodes.csv', 'r') as nodecsv:
      reader = csv.reader(nodecsv)
      data = [n for n in reader]
      # Get the first line in the csv as the header.
      nodes_header = data[0]
      nodes = data[1:]
  # Store the node IDs (the first item in each row)
  node_ids = [n[0] for n in nodes]

  # Read the edges csv file
  with open('data/edges.csv', 'r') as edgecsv:
      reader = csv.reader(edgecsv)
      edges = [tuple(e) for e in reader][1:]

  # Construct the graph
  G = nx.Graph()
  G.add_nodes_from(node_ids)
  G.add_weighted_edges_from(edges, weight="distance")
  print(nx.info(G))

  attributes = createAttributeDictionaries(nodes_header, nodes)

  # Set node attributes
  for key in attributes:
    nx.set_node_attributes(G, attributes[key], key)
  
  for i in range(5):
    print("next round")
    for node in G.nodes:
      node_attributes = G.nodes[node]
      for neighbor in G.neighbors(node):
        neighbor_attributes = G.nodes[neighbor]
        distance = float(G[node][neighbor]["distance"])
        [min_x, min_y, max_x, max_y] = bbox_intersection(node_attributes["min_x"]-distance, node_attributes["min_y"]-distance, node_attributes["max_x"]+distance, node_attributes["max_y"]+distance, neighbor_attributes["min_x"], neighbor_attributes["min_y"], neighbor_attributes["max_x"], neighbor_attributes["max_y"])
        print(neighbor)
        print([min_x, min_y, max_x, max_y])
        if (max_x is not None):
          neighbor_attributes["min_x"] = min_x
          neighbor_attributes["min_y"] = min_y
          neighbor_attributes["max_x"] = max_x
          neighbor_attributes["max_y"] = max_y
  
if __name__ == '__main__':
  main()
