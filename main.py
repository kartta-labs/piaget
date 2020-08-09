# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Piaget is a system to geotag historical building photos.
"""

import csv
import json
from operator import itemgetter
import networkx
import numpy as np
from scipy.stats import multivariate_normal, truncnorm
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import re
from scipy import signal


X = 1; Y= 0

parser = argparse.ArgumentParser(description="Piaget: Enter the input JSON file using the --experiments flag.")
required = parser.add_argument_group('required arguments')
required.add_argument("--experiments", help="path to the experiments json file.", required=True)


def get_valid_filename(s):
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


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

  def get_area(self):
    return self.dx * self.dy * self.nx * self.ny

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
  
  @staticmethod
  def haversine_distance(lon1, lat1, lon2, lat2):
      """
      Calculate the distance between two points on the earth given
      their latitude and longitude in decimal degrees.
      """
      #degrees to radians:
      lon1 = np.radians(lon1)
      lat1 = np.radians(lat1)
      lon2 = np.radians(lon2)
      lat2 = np.radians(lat2)

      delta_lon = np.subtract(lon2, lon1)
      delta_lat = np.subtract(lat2, lat1)

      a = np.sin((delta_lat)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((delta_lon)/2)**2
      c = 2*np.arcsin(np.sqrt(a))
      earth_radius_meters = 6371 * 1000

      return earth_radius_meters*c

  def distance_grid(self, pos):
    if pos.tobytes() not in self.distance_grid_cache:
      distance_grid = self.haversine_distance(self.x, self.y, pos[X], pos[Y])
      # distance_grid = np.sqrt((pos[X] - self.x)**2 + (pos[Y] - self.y)**2)
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

  def get_probability_in_cells(self):
    return self.grid.cells * self.grid.dy * self.grid.dx

  def get_truncnorm_grid_old(self, pos, a, loc, scale):
    key = np.array([pos[0], pos[1], a, loc, scale])
    if key.tobytes() not in self.truncnorm_grid_cache:
      self.truncnorm_grid_cache[key.tobytes()] = truncnorm.pdf(self.grid.distance_grid(pos), a, b = np.inf, loc = loc, scale = scale)
    return self.truncnorm_grid_cache[key.tobytes()]

  def get_mixed_uniform_grid(self, pos, prob):
    grid = Grid(bbox=self.grid.bbox, ny=self.grid.ny, nx=self.grid.nx)
    grid.cells *= (1-prob)
    row, col = grid.get_pos_index(pos)
    grid.cell[row][col] = prob
    
  def get_truncnorm_grid(self, pos, a, loc, scale):
    key = np.array([a, loc, scale])
    if key.tobytes() not in self.truncnorm_lookup_grid_cache:
      truncnorm_lookup_grid = Grid(bbox=Rectangle(bottom_left=Point(self.grid.bbox.bottom_left.y - (self.grid.ny-1) * self.grid.dy, self.grid.bbox.bottom_left.x - (self.grid.nx-1) * self.grid.dx), top_right=self.grid.bbox.top_right), ny=self.grid.ny*2-1, nx=self.grid.nx*2-1) 
      dist = truncnorm_lookup_grid.distance_grid(self.grid.pos[0,0])
      dist[dist==0] = np.finfo(float).eps
      dist_scaled = (dist - loc) / scale 
      self.truncnorm_lookup_grid_cache[key.tobytes()] = truncnorm.pdf(dist_scaled, a = a, b = np.inf) / (scale *dist)
    row, col = self.grid.get_pos_index(pos)
    return self.truncnorm_lookup_grid_cache[key.tobytes()][self.grid.ny-1 - row:self.grid.ny-1 - row + self.grid.ny, self.grid.nx-1 - col:self.grid.nx-1 - col + self.grid.nx]

  def get_full_truncnorm_grid(self, a, loc, scale):
    key = np.array([a, loc, scale])
    if key.tobytes() not in self.truncnorm_lookup_grid_cache:
      dist = self.grid.distance_grid(self.grid.pos[0,0])
      dist[dist==0] = np.finfo(float).eps
      dist_scaled = (dist - loc) / scale
      top_right_slice = truncnorm.pdf(dist_scaled, a = a, b = np.inf) / (scale * dist)
      lookup_grid = np.concatenate((np.fliplr(top_right_slice)[:,:-1],top_right_slice),axis=1)
      lookup_grid = np.concatenate((np.flipud(lookup_grid)[:-1,:], lookup_grid), axis=0)
      
      self.truncnorm_lookup_grid_cache[key.tobytes()] = lookup_grid 
    return self.truncnorm_lookup_grid_cache[key.tobytes()]


class Geotagging(object):
  def __init__(self, bbox: Rectangle, ny, nx, path_to_nodes_csv = None, path_to_edges_csv = None, nodes_csv_list = None, edges_csv_list = None):
    self.graph = networkx.Graph()
    self.bbox = bbox
    self.nx = nx
    self.ny = ny

    self.load_nodes(path_to_nodes_csv, nodes_csv_list)
    self.load_edges(path_to_edges_csv, edges_csv_list)

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

  def load_nodes(self, path_to_csv = None, nodes_csv_list = None):
    # Read the nodes csv file
    if path_to_csv is not None:
      nodes_csv_list = open(path_to_csv, "r")
    reader = csv.reader(nodes_csv_list)
    data = [n for n in reader]
    # Get the first line in the csv as the header.
    self.nodes_header = data[0]
    self.nodes = data[1:]
    # Store the node IDs (the first item in each row)
    self.node_ids = [n[0] for n in self.nodes]
    self.graph.add_nodes_from(self.node_ids)
    self.set_attributes()

  def load_edges(self, path_to_csv = None, edges_csv_list = None):
    # Read the edges csv file
    if path_to_csv is not None:
      edges_csv_list = open(path_to_csv, "r")
    reader = csv.reader(edges_csv_list)
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
    propogate_time = time.time()
    at_least_one_probability_was_updated = False
    for node in self.graph.nodes:
      # print("node: {}".format(node))
      node_attributes = self.graph.nodes[node]
      for neighbor in self.graph.neighbors(node):
        neighbor_attributes = self.graph.nodes[neighbor]
        [mean, std_or_prob] =  list(map(float, self.graph[node][neighbor]["mean_distance_and_standard_deviation"]))
        # print("neighbor: {}".format(neighbor))
        edge_message = [node, neighbor]
        edge_message.sort()
        edge_message = ";".join(edge_message)
        if mean == 0:
          # if it is a sameness edge.
          node_attributes["probability_grid"].normalize()
          likelihood = (1 - node_attributes["probability_grid"].get_probability_in_cells()) * (1-std_or_prob)/(node_attributes["probability_grid"].grid.nx*node_attributes["probability_grid"].grid.ny-1) + node_attributes["probability_grid"].get_probability_in_cells() * std_or_prob
        else:
          a = (0 - mean) / std_or_prob
          # if it is a neighborhood edge.
          likelihood = np.zeros((neighbor_attributes["probability_grid"].grid.ny, neighbor_attributes["probability_grid"].grid.nx), dtype=float)
          
          # following is the old calculation of the likelihood, before I figured out it can be done
          # by convolve2d. Keeping it for the record ...
          #for col in node_attributes["probability_grid"].grid.pos:
          #  for pos in col:
          #    likelihood += node_attributes["probability_grid"].get_probability_in_cell(pos[Y], pos[X]) * neighbor_attributes["probability_grid"].get_truncnorm_grid(pos = pos, a = a, loc = mean, scale = std_or_prob)
          likelihood =  signal.convolve2d(node_attributes["probability_grid"].get_probability_in_cells(),neighbor_attributes["probability_grid"].get_full_truncnorm_grid(a = a, loc = mean, scale = std_or_prob),mode="valid")

        if (not node_attributes["received_message_sources"].issubset(neighbor_attributes["received_message_sources"]) or edge_message not in neighbor_attributes["received_message_sources"]):
          # if no new message source is available, skip.
          neighbor_attributes["received_message_sources"].update(node_attributes["received_message_sources"])
          neighbor_attributes["received_message_sources"].add(edge_message)
          #if np.sum(likelihood) == 0:
            #print("likelihood was zero for node {} and neighbor {}".format(node,neighbor))
          if (neighbor_attributes["locked"].lower() != "true" and np.sum(likelihood) > 0):
            # if the neighbor"s location is locker, skip.
            # if, for any reason, likelihood is zero, skip.
            likelihood /= np.sum(likelihood)
            neighbor_attributes["probability_grid"].posterior(likelihood, True)
            at_least_one_probability_was_updated = True
    return at_least_one_probability_was_updated

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
  args = parser.parse_args()
  with open(args.experiments, "r") as stream:
    experiments = json.load(stream)

  experiments_path = Path(args.experiments)
  results = {}

  for configs in experiments["experiments"]:
    start_time = time.time()
    results[configs["id"]] = {}
    results[configs["id"]]["data"] = []

    bottom_left = Point(configs["bottom_left_y"], configs["bottom_left_x"])
    top_right = Point(configs["top_right_y"], configs["top_right_x"])
    nx = configs["nx"]
    ny = configs["ny"]
    bbox = Rectangle(bottom_left, top_right)
    
    geotagging = Geotagging(bbox=bbox, ny=ny, nx=nx, nodes_csv_list=configs["nodes"],edges_csv_list=configs["edges"])
    print(networkx.info(geotagging.graph))

    counter = 0
    while(True):
      print("round {}".format(counter))
      counter += 1
      if(not geotagging.propogate()):
        break
    
    plt.ioff()
    """
    for node in geotagging.graph.nodes:
      node_attributes = geotagging.graph.nodes[node]
      plt.close()
      plt.imshow(node_attributes["probability_grid"].grid.cells, cmap="viridis", origin="lower",extent=[bottom_left.x, top_right.x, bottom_left.y, top_right.y])
      plt.colorbar()
      plt.savefig("{}/{}.png".format(experiments_path.parent,get_valid_filename(node)))
      node_attributes["probability_grid"].normalize()
      print("This node:")
      print(node)
      max_pos_index = (np.unravel_index(node_attributes["probability_grid"].grid.cells.argmax(), node_attributes["probability_grid"].grid.cells.shape))
      print(max_pos_index)
      print(node_attributes["probability_grid"].grid.pos[max_pos_index])
      print(node_attributes["received_message_sources"])
    """
    average_distance_of_max_to_ground_truth = 0
    average_expected_distance_to_ground_truth = 0
    count = 0
    for node in geotagging.graph.nodes:
      count += 1
      node_attributes = geotagging.graph.nodes[node]
      node_attributes["probability_grid"].normalize()
      node_attributes["expected_distance_to_ground_truth"] = np.sum(node_attributes["probability_grid"].get_probability_in_cells() * node_attributes["probability_grid"].grid.distance_grid(np.array([node_attributes["mean_y"], node_attributes["mean_x"]])))
      average_expected_distance_to_ground_truth += node_attributes["expected_distance_to_ground_truth"] 
      max_pos_index = (np.unravel_index(node_attributes["probability_grid"].grid.cells.argmax(), node_attributes["probability_grid"].grid.cells.shape))
      node_attributes["max_probablity_cell_center"] = list(node_attributes["probability_grid"].grid.pos[max_pos_index])
      node_attributes["distance_of_max_to_ground_truth"] = Grid.haversine_distance(node_attributes["mean_x"], node_attributes["mean_y"], node_attributes["max_probablity_cell_center"][X], node_attributes["max_probablity_cell_center"][Y])
      average_distance_of_max_to_ground_truth += node_attributes["distance_of_max_to_ground_truth"] 
      node_attributes["probability_grid"] = node_attributes["probability_grid"].grid.cells
      node_attributes["received_message_sources"] = list(node_attributes["received_message_sources"])
      node_attributes["id"] = node
      results[configs["id"]]["data"].append(node_attributes)
    results[configs["id"]]["summary"] = {
      "average_distance_of_max_to_ground_truth": average_distance_of_max_to_ground_truth/count,
      "average_expected_distance_to_ground_truth": average_expected_distance_to_ground_truth/count,
      "time": time.time() - start_time,
      "rounds": counter,
      "networkx.info": networkx.info(geotagging.graph),
      "number_of_nodes": geotagging.graph.number_of_nodes(),
      "number_of_edges": geotagging.graph.number_of_edges(), 
      "number_of_connected_components": networkx.number_connected_components(geotagging.graph)
      }
    print("summary:")  
    print(results[configs["id"]]["summary"])
    print("finished in {} rounds.".format(counter))
    print(time.time() - start_time)
    
    for edge in geotagging.graph.edges:
      mean_distance_and_standard_deviation = geotagging.graph.edges[edge]["mean_distance_and_standard_deviation"]
      geotagging.graph.edges[edge]["mean_distance_and_standard_deviation"] = float(mean_distance_and_standard_deviation[0])**10
    networkx.write_edgelist(geotagging.graph, path="graph.edgelist", delimiter=";")
    H = networkx.read_edgelist(path="graph.edgelist", delimiter=";")
    networkx.draw(H)
    plt.show()
    plt.savefig("graph.png")

  with open("{}/results-{}-from-{}-to-{}.json".format(experiments_path.parent, experiments_path.stem, experiments["experiments"][0]["id"], experiments["experiments"][-1]["id"]), "w") as outfile:
    json.dump(results, outfile, cls=NumpyEncoder, sort_keys=True, indent=2)
    
if __name__ == "__main__":
  main()
