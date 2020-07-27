# Lint as: python3
"""TODO(sasantv): DO NOT SUBMIT without one-line documentation for main.

TODO(sasantv): DO NOT SUBMIT without a detailed description of main.
NOTE: Everywhere in this code Y is index 0, and X is index 1.
"""
import csv
from operator import itemgetter
import networkx
import numpy as np
from scipy.stats import multivariate_normal, truncnorm
import matplotlib.pyplot as plt
import time
import argparse
import yaml

X = 1; Y= 0

parser = argparse.ArgumentParser(description="Piaget: Enter the input YAML file using the --configs flag.")
required = parser.add_argument_group('required arguments')
required.add_argument("--configs", help="path to the configs YAML file.", required=True)


class Point(object):
  """A 2D point in xy plane"""
  def __init__(self, y: float, x: float):
    self.y = y
    self.x = x

class Rectangle(object):
  """Rectangle defined by its bottom_left and top_right corners"""
  def __init__(self, bottom_left: Point, top_right: Point):
    self.bottom_left = bottom_left
    self.top_right = top_right


class Grid(object):
  """A grid of cells. The grid is defined by a boundingbox (bbox) and the number of cells in each direction (nx, ny)"""
  def __init__(self, bbox: Rectangle, ny, nx):
    """Creates a grid given a bbox and number of cells in each direction"""
    self.bbox = bbox
    self.nx = nx
    self.ny = ny
    self.dx = float(bbox.top_right.x - bbox.bottom_left.x)/float(nx)
    self.dy = float(bbox.top_right.y - bbox.bottom_left.y)/float(ny)
    self.distance_grid_cache = {}
    self.construct_grid()
  
  def construct_grid(self):
    # Create an offset to center the cells.
    x_offset = self.dx/2
    y_offset =  self.dy/2
    self.y, self.x = np.mgrid[self.bbox.bottom_left.y + y_offset:self.bbox.top_right.y:self.dy, self.bbox.bottom_left.x + x_offset:self.bbox.top_right.x:self.dx]
    self.pos = np.dstack((self.y, self.x))
    self.cells = np.ones((self.ny, self.nx), dtype=float)

  def get_cell(self, y, x):
    row = int((y - self.bbox.bottom_left.y)/self.dy)
    col = int((x - self.bbox.bottom_left.x)/self.dx)
    return self.cells[row][col]

  def get_pos_index(self, pos):
    row = int((pos[Y] - self.bbox.bottom_left.y)/self.dy)
    col = int((pos[X] - self.bbox.bottom_left.x)/self.dx)
    return (row, col)

  def set_cell(self, y, x, value):
    row_index = int((y - self.bbox.bottom_left.y)/self.dy)
    col_index = int((x - self.bbox.bottom_left.x)/self.dx)
    self.cells[row_index][col_index] = value
  
  def haversine_distance(lon1, lat1, lon2, lat2):
      """
      Calculate the distance between two points on the earth given
      their latitude and longitude in decimal degrees.
      """
      #degrees to radians:
      lon1 = np.radians(lon1.values)
      lat1 = np.radians(lat1.values)
      lon2 = np.radians(lon2.values)
      lat2 = np.radians(lat2.values)

      delta_lon = np.subtract(lon2, lon1)
      delta_lat = np.subtract(lat2, lat1)

      a = np.sin((delta_lat)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((delta_lon)/2)**2
      c = 2*np.arcsin(np.sqrt(a))
      earth_radius_meters = 6371 * 1000

      return earth_radius_meters*c

  def distance_grid(self, pos):
    if pos.tobytes() not in self.distance_grid_cache:
      # distance_grid = self.haversine_distance(self.x, self.y, pos[X], pos[Y])
      distance_grid = np.sqrt((pos[X] - self.x)**2 + (pos[Y] - self.y)**2)
      self.distance_grid_cache[pos.tobytes()] = distance_grid
    return self.distance_grid_cache[pos.tobytes()]

class ProbabilityGrid(object):
  def __init__(self, bbox: Rectangle, ny, nx, normalize: bool = True):
    self.grid = Grid(bbox=bbox, ny=ny, nx=nx)
    self.truncnorm_grid_cache = {}  # deprecated
    self.truncnorm_lookup_grid_cache = {}
    if normalize:
      self.normalize()

  def overwrite_with_normal_dist(self, mean, cov, normalize: bool = True):
    rv = multivariate_normal(mean=mean, cov=cov)
    self.grid.cells = rv.pdf(self.grid.pos)
    if normalize:
      self.normalize
  def probability_sum_check(self):
    return np.sum(self.grid.cells) * (self.grid.dy * self.grid.dx)

  def normalize(self):
    norm = self.probability_sum_check()
    if norm == 0:
        norm = np.finfo(self.grid.cells.dtype).eps
    self.grid.cells /= norm
  
  def posterior(self, likelihood, normalize: bool = True):
    self.grid.cells *= likelihood
    if normalize:
      self.normalize()

  def get_probability_in_cell(self, y, x):
    return self.grid.get_cell(y, x) * self.grid.dy * self.grid.dx
 
  def get_truncnorm_grid_old(self, pos, a, loc, scale):
    key = np.array([pos[0], pos[1], a, loc, scale])
    if key.tobytes() not in self.truncnorm_grid_cache:
      self.truncnorm_grid_cache[key.tobytes()] = truncnorm.pdf(self.grid.distance_grid(pos), a, b = np.inf, loc = loc, scale = scale)
    return self.truncnorm_grid_cache[key.tobytes()]

  def get_truncnorm_grid(self, pos, a, loc, scale):
    key = np.array([a, loc, scale])
    if key.tobytes() not in self.truncnorm_lookup_grid_cache:
      truncnorm_lookup_grid = Grid(bbox=Rectangle(bottom_left=Point(self.grid.bbox.bottom_left.y - (self.grid.ny-1) * self.grid.dy, self.grid.bbox.bottom_left.x - (self.grid.nx-1) * self.grid.dx), top_right=self.grid.bbox.top_right), ny=self.grid.ny*2-1, nx=self.grid.nx*2-1) 
      self.truncnorm_lookup_grid_cache[key.tobytes()] = truncnorm.pdf(truncnorm_lookup_grid.distance_grid(self.grid.pos[0,0]), a = a, b = np.inf, loc = loc, scale = scale) / truncnorm_lookup_grid.distance_grid(self.grid.pos[0,0])
    row, col = self.grid.get_pos_index(pos)
    return self.truncnorm_lookup_grid_cache[key.tobytes()][self.grid.ny-1 - row:self.grid.ny-1 - row + self.grid.ny, self.grid.nx-1 - col:self.grid.nx-1 - col + self.grid.nx]

class Geotagging(object):
  def __init__(self, path_to_nodes_csv, path_to_edges_csv, bbox: Rectangle, ny, nx):
    self.graph = networkx.Graph()
    self.bbox = bbox
    self.nx = nx
    self.ny = ny

    self.load_nodes(path_to_nodes_csv)
    self.load_edges(path_to_edges_csv)

  def set_attributes(self):
    header_to_index = {}
    for i in range(0, len(self.nodes_header)):
      header_to_index[self.nodes_header[i]] = i
    
    # get mean_x and mean_y attributes
    attributes_dict = {}
    for key in ["mean_y", "mean_x"]:
      attributes = {}
      for node in self.nodes:
        attributes[node[header_to_index["id"]]] = float(node[header_to_index[key]])
      attributes_dict[key] = attributes

    for key in ["locked"]:
      attributes = {}
      for node in self.nodes:
        attributes[node[header_to_index["id"]]] = node[header_to_index[key]]
      attributes_dict[key] = attributes

    attributes = {}
    for node in self.nodes:
      covariance = np.array([[float(node[header_to_index["cov_yy"]]), float(node[header_to_index["cov_yx"]])], [float(node[header_to_index["cov_yx"]]), float(node[header_to_index["cov_xx"]])]])
      mean = [float(node[header_to_index["mean_y"]]), float(node[header_to_index["mean_x"]])]
      prob_grid = ProbabilityGrid(bbox=self.bbox, ny=self.ny, nx=self.nx, normalize=True)
      if node[header_to_index["known_location"]].lower() == "true":
        prob_grid.overwrite_with_normal_dist(mean=mean, cov=covariance, normalize=True)
      attributes[node[header_to_index["id"]]] = prob_grid
    attributes_dict["probability_grid"] = attributes

    attributes = {}
    for node in self.nodes:
      attributes[node[header_to_index["id"]]] = set()
      if node[header_to_index["known_location"]].lower() == "true":
        attributes[node[header_to_index["id"]]].add(node[header_to_index["id"]])
    attributes_dict["received_message_sources"] = attributes


    for key in attributes_dict:
      networkx.set_node_attributes(self.graph, attributes_dict[key], key)

  def load_nodes(self, path_to_csv):
    # Read the nodes csv file
    with open(path_to_csv, "r") as nodecsv:
        reader = csv.reader(nodecsv)
        data = [n for n in reader]
        # Get the first line in the csv as the header.
        self.nodes_header = data[0]
        self.nodes = data[1:]
    # Store the node IDs (the first item in each row)
    self.node_ids = [n[0] for n in self.nodes]
    self.graph.add_nodes_from(self.node_ids)
    self.set_attributes()

  def load_edges(self, path_to_csv):
    # Read the edges csv file
    with open(path_to_csv, "r") as edgecsv:
      reader = csv.reader(edgecsv)
      data = [n for n in reader]
      # edges = [tuple(e) for e in reader][1:]
      self.edges = []
      for e in data[1:]:
        self.edges.append(
          tuple(
              [e[0],e[1],np.array([e[2],e[3]])]
            )
        )
    self.graph.add_weighted_edges_from(self.edges, weight="mean_distance_and_standard_deviation")

  def propogate(self) -> bool:
    at_least_one_probability_was_updated = False
    start_round = time.time()
    for node in self.graph.nodes:
      print("node: {}".format(node))
      node_attributes = self.graph.nodes[node]
      for neighbor in self.graph.neighbors(node):
        start_neighbor = time.time()
        neighbor_attributes = self.graph.nodes[neighbor]
        [mean, std] =  list(map(float, self.graph[node][neighbor]["mean_distance_and_standard_deviation"]))
        a = (0 - mean) / std
        print("neighbor: {}".format(neighbor))
        likelihood = np.zeros((neighbor_attributes["probability_grid"].grid.ny, neighbor_attributes["probability_grid"].grid.nx), dtype=float)
        for col in node_attributes["probability_grid"].grid.pos:
          for pos in col:
            start_pos = time.time()
            # TODO: following is not correct. it needs to be adjusted for the circle perim
            likelihood += node_attributes["probability_grid"].get_probability_in_cell(pos[Y], pos[X]) * neighbor_attributes["probability_grid"].get_truncnorm_grid(pos = pos, a = a, loc = mean, scale = std)
            # print("time for one pos: {}".format(time.time() - start_pos))
        if (not node_attributes["received_message_sources"].issubset(neighbor_attributes["received_message_sources"])):
          # if no new message source is available, skip.
          neighbor_attributes["received_message_sources"].update(node_attributes["received_message_sources"])
          if (neighbor_attributes["locked"].lower() != "true" and np.sum(likelihood) > 0):
            # if the neighbor"s location is locker, skip.
            # if, for any reason, likelihood is zero, skip.
            likelihood /= np.sum(likelihood)
            neighbor_attributes["probability_grid"].posterior(likelihood, True)
            at_least_one_probability_was_updated = True
    return at_least_one_probability_was_updated


def main():

  args = parser.parse_args()
  print(args.configs)
  with open(args.configs, "r") as stream:
    configs = yaml.safe_load(stream)
  

  experiment_folder="manhattan"
  bottom_left = Point(-2,-2)
  top_right = Point(5,5)
  nx = configs["nx"]
  ny = configs["ny"]
  bbox = Rectangle(bottom_left, top_right)
  
  geotagging = Geotagging("data/{}/nodes.csv".format(experiment_folder), "data/{}/edges.csv".format(experiment_folder), bbox=bbox, ny=ny, nx=nx)
  print(networkx.info(geotagging.graph))

  rounds = 10
  for i in range(rounds):
    print("round {}".format(i))
    if(not geotagging.propogate()):
      break


  print("results:")
  plt.ioff()
  for node in geotagging.graph.nodes:
    node_attributes = geotagging.graph.nodes[node]
    plt.close()
    plt.imshow(node_attributes["probability_grid"].grid.cells, cmap="viridis", origin="lower",extent=[bottom_left.x, top_right.x, bottom_left.y, top_right.y])
    plt.colorbar()
    plt.savefig("data/{}/{}.png".format(experiment_folder,node))
    np.max(node_attributes["probability_grid"].normalize())
    print("This node:")
    print(node)
    # print(node_attributes["probability_grid"].grid.cells)

    max_pos_index = (np.unravel_index(node_attributes["probability_grid"].grid.cells.argmax(), node_attributes["probability_grid"].grid.cells.shape))
    print(max_pos_index)
    print(node_attributes["probability_grid"].grid.pos[max_pos_index])
    print(node_attributes["received_message_sources"])
  
  print("finished in {} rounds.".format(i+1))

if __name__ == "__main__":
  main()
