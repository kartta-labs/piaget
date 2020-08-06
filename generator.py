
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

import math
import csv
from operator import itemgetter
import random
import json
from pathlib import Path
import argparse
import uuid
import numpy as np

parser = argparse.ArgumentParser(description="Enter the input JSON configs using the --configs flag.")
required = parser.add_argument_group('required arguments')
required.add_argument("--configs", help="path to the configs json file.", required=True)


class MapFeatures(object):
  def __init__(self, path_to_features_json, years):
    self.years = years
    self.years_dict = {}
    self.data_dict = {}
    self.street_dict = {}
    self.nodes = {}
    self.edges = {}

    self.read_input(path_to_features_json)
    self.create_years_dict()
    self.create_address_dict()
    self.sort_housenumbers()
    self.create_all_nodes()
    self.create_all_edges()

  def read_input(self, path_to_features_json):
    with open(path_to_features_json) as json_file:
      self.raw_data = json.load(json_file)["data"]

  def create_years_dict(self):
    for year in self.years:
      self.years_dict[year] = []
    for feature in self.raw_data:
      if all (key in feature["properties"] for key in ["building","addr:street","addr:housenumber"]):
        self.data_dict[feature["id"]] = feature
        for year in self.years:
          if self.existed_in_year(feature, year):
            self.years_dict[year].append(feature)
          
  def existed_in_year(self, feature, year):
    if "properties" not in feature:
      return True
    # if all (key in feature["properties"] for key in ["start_date","end_date"]):
    if "start_date" not in feature["properties"]:
      if "end_date" not in feature["properties"]:
        return True
      return int(feature["properties"]["end_date"]) > year
    else:
      if int(feature["properties"]["start_date"]) > year:
        return False
      else:
        if "end_date" not in feature["properties"]:
          return True
        return int(feature["properties"]["end_date"]) > year

    return True

  def create_address_dict(self):
    for year in self.years:
      if year not in self.street_dict:
        self.street_dict[year] = {}
      for feature in self.years_dict[year]:
        if feature["properties"]["addr:street"].lower() not in self.street_dict[year]:
          self.street_dict[year][feature["properties"]["addr:street"].lower()] = {"odd":[], "even":[]}
        housenumber = float(feature["properties"]["addr:housenumber"])
        odd_or_even_key = "even" if housenumber % 2 == 0 else "odd"
        self.street_dict[year][feature["properties"]["addr:street"].lower()][odd_or_even_key].append((housenumber, feature["id"]))
  
  def sort_housenumbers(self):
    for year in self.years:
      for street in self.street_dict[year]:
        for odd_or_even_key in self.street_dict[year][street]:
          self.street_dict[year][street][odd_or_even_key].sort(key=itemgetter(0))

  def create_all_edges(self):
    for year in self.street_dict:
      self.edges[year] = {}
      for street in self.street_dict[year]:
        for odd_or_even_key in self.street_dict[year][street]:
          housenumbers = self.street_dict[year][street][odd_or_even_key]
          for i in range(len(housenumbers)-1):
            id_1 = housenumbers[i][1]
            id_2 = housenumbers[i+1][1]
            edge = (id_1, id_2)
            self.edges[year][edge] = self.haversine_distance(self.nodes[year][id_1], self.nodes[year][id_2])

  def create_all_nodes(self):
    for year in self.street_dict:
      self.nodes[year] = {}
      for street in self.street_dict[year]:
        for odd_or_even_key in self.street_dict[year][street]:
          housenumbers = self.street_dict[year][street][odd_or_even_key]
          for i in range(len(housenumbers)):
            id = housenumbers[i][1]
            self.nodes[year][id] = self.cacluate_surface_centroid(self.data_dict[id]["geometry"]["coordinates"][0])

  def get_all_nodes_without_year_grouping(self):
    return list(self.data_dict.keys())

  def get_all_nodes(self):
    return self.nodes

  def get_all_edges(self):
    return self.edges

  def cacluate_surface_centroid(self, points):
    area = 0; lon = 0; lat = 0
    points_count = len(points)-1
    for i in range(points_count):
      lon += points[i][0]
      lat += points[i][1]
    
    return [lon/points_count, lat/points_count]

    # TODO: The following is buggy, not sure why. So we just return the mean.
    for i in range(len(points)-1):
      p1 = points[i]
      p2 = points[i+1]
      coefficient = p1[0] * p2[1] - p2[0] * p1[1]
      area += coefficient / 2.0
      lon += (p1[0] + p2[0]) * coefficient
      lat += (p1[1] + p2[1]) * coefficient

    coefficient = area * 6
    return [lon / coefficient, lat / coefficient]

  @staticmethod
  def haversine_distance(point1, point2):
      """
      Calculate the distance between two points on the earth given
      their latitude and longitude in decimal degrees.
      """
      #degrees to radians:
      lon1 = math.radians(point1[0])
      lat1 = math.radians(point1[1])
      lon2 = math.radians(point2[0])
      lat2 = math.radians(point2[1])

      delta_lon = lon2 - lon1
      delta_lat = lat2 - lat1

      a = math.sin((delta_lat)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((delta_lon)/2)**2
      c = 2*math.asin(math.sqrt(a))
      earth_radius_meters = 6371 * 1000

      return earth_radius_meters*c

  @staticmethod
  def generate_random_indicies(seed, count, maximum):
    if count == 0:
      return []
    if count > maximum:
      return [n for n in range(maximum + 1)]
    indicies = set()
    reverse = False
    if count > int(maximum/2):
      count = maximum + 1 - count
      reverse = True
    random.seed(seed)
    while (True):
      indicies.add(random.randint(0, maximum))
      if len(indicies) == count:
        break

    if reverse:
      all_indicies = set([n for n in range(maximum + 1)])
      indicies = all_indicies.difference(indicies)
    return list(indicies)

def main():
  args = parser.parse_args()
  with open(args.configs, "r") as stream:
    configs = json.load(stream)
    configs_path = Path(args.configs)

  payload = {"experiments":[]}

  for experiment in configs["experiments"]:
    node_to_neighbors = {}
    years = experiment["years"]
    map = MapFeatures(experiment["path_to_features_json"], years)
    experiment["nodes"] = ["id,mean_y,mean_x,known_location,locked,cov_yy,cov_yx,cov_xx,year"]
    experiment["edges"] = ["source,target,mean_distance,standard_deviation,year"]

    for i in range(len(years)):
      year = years[i]
      number_of_unique_photos = experiment["number_of_unique_photos_in_year"][i]
      ratio_of_sameness = experiment["ratio_of_sameness_in_year"][i]
      ratio_of_seeds = experiment["ratio_of_seeds_in_year"][i]
      ratio_of_seeds_in_year_among_sameness = experiment["ratio_of_seeds_in_year_among_sameness"][i]
      ratio_of_false_matches_in_year =  experiment["ratio_of_false_matches_in_year"][i]
      seed_cov_xx_and_cov_yy_degrees = experiment["seed_cov_xx_and_cov_yy_degrees"]


      node_to_neighbors[year] = {}
      edges = map.get_all_edges()[year]
      for edge in edges:
        node_to_neighbors[year].setdefault(edge[0], set()).add(edge[1])
        node_to_neighbors[year].setdefault(edge[1], set()).add(edge[0])

      nodes = map.get_all_nodes()[year]
      node_keys = list(nodes.keys())
      random_indicies = MapFeatures.generate_random_indicies(experiment["randomness_seed"], number_of_unique_photos, len(node_keys)-1)
      random_indicies_of_seeds = MapFeatures.generate_random_indicies(experiment["randomness_seed"], int(number_of_unique_photos*ratio_of_seeds), len(random_indicies)-1)
      selected_nodes = set()
      for i in range(len(random_indicies)):
        node_key_index = random_indicies[i]
        node = node_keys[node_key_index]
        photo_id = str(uuid.uuid4())
        node_photo_id = node + ":" + photo_id
        selected_nodes.add(node_photo_id)
        seed = i in random_indicies_of_seeds
        experiment["nodes"].append(",".join("{}".format(n) for n in [node_photo_id, nodes[node][1], nodes[node][0],seed,seed,seed_cov_xx_and_cov_yy_degrees,0,seed_cov_xx_and_cov_yy_degrees,year]))
      neighbors_of_selected_nodes = []
      for node_photo_id in sorted(selected_nodes):
        node = node_photo_id.split(":")[0]
        photo_id = node_photo_id.split(":")[1]
        for neighbor in sorted(node_to_neighbors[year][node]):
          neighbor_id = neighbor + ":" + photo_id
          neighbors_of_selected_nodes.append(neighbor_id)
          edge = (node, neighbor)
          edge_id = (node_photo_id, neighbor_id)
          if edge not in edges:
            edge = (neighbor, node)
            edge_id = (neighbor_id, node_photo_id)
          distance = edges[edge]
          if experiment["add_noise_to_mean_distance"].lower() == "true":
            distance += np.random.normal(0, experiment["std"], 1)[0]
            distance = max(0,distance)
          experiment["edges"].append(",".join("{}".format(n) for n in [edge_id[0], edge_id[1],distance,experiment["std"],year]))
          experiment["nodes"].append(",".join("{}".format(n) for n in [neighbor_id, nodes[neighbor][1], nodes[neighbor][0],False,False,seed_cov_xx_and_cov_yy_degrees,0,seed_cov_xx_and_cov_yy_degrees,year]))

      # add sameness nodes
      random_indicies = MapFeatures.generate_random_indicies(experiment["randomness_seed"], int(number_of_unique_photos*ratio_of_sameness), len(neighbors_of_selected_nodes)-1)
      random_indicies_of_seeds = MapFeatures.generate_random_indicies(experiment["randomness_seed"], int(len(random_indicies)*ratio_of_seeds_in_year_among_sameness), len(random_indicies)-1)
      selected_nodes = set()
      for i in range(len(random_indicies)):
        node_key_index = random_indicies[i]
        node_photo_id = neighbors_of_selected_nodes[node_key_index]
        node = node_photo_id.split(":")[0]
        new_photo_id = str(uuid.uuid4())
        node_new_id = node + ":" + new_photo_id
        experiment["edges"].append(",".join("{}".format(n) for n in [node_photo_id, node_new_id,0,experiment["true_matching_confidence"],year]))
        selected_nodes.add(node_new_id)
        seed = i in random_indicies_of_seeds
        experiment["nodes"].append(",".join("{}".format(n) for n in [node_new_id, nodes[node][1], nodes[node][0],seed,seed,seed_cov_xx_and_cov_yy_degrees,0,seed_cov_xx_and_cov_yy_degrees,year]))
      for node_photo_id in selected_nodes:
        node = node_photo_id.split(":")[0]
        photo_id = node_photo_id.split(":")[1]
        for neighbor in node_to_neighbors[year][node]:
          neighbor_id = neighbor + ":" + photo_id
          edge = (node,neighbor)
          edge_id = (node_photo_id, neighbor_id)
          if edge not in edges:
            edge = (neighbor,node)
            edge_id = (neighbor_id, node_photo_id)
          distance = edges[edge]
          if experiment["add_noise_to_mean_distance"].lower() == "true":
            distance += np.random.normal(0, experiment["std"], 1)[0]
            distance = max(0,distance)
          experiment["edges"].append(",".join("{}".format(n) for n in [edge_id[0], edge_id[1],distance,experiment["std"],year]))
          experiment["nodes"].append(",".join("{}".format(n) for n in [neighbor_id, nodes[neighbor][1], nodes[neighbor][0],False,False,seed_cov_xx_and_cov_yy_degrees,0,seed_cov_xx_and_cov_yy_degrees,year]))

      # add false matching
      random_indicies = MapFeatures.generate_random_indicies(experiment["randomness_seed"], int(2 * number_of_unique_photos*ratio_of_false_matches_in_year), len(neighbors_of_selected_nodes)-1)
      for i in range(int(len(random_indicies)/2)):
        node_key_index_1 = random_indicies[i]
        node_photo_id_1 = neighbors_of_selected_nodes[node_key_index_1]
        node_key_index_2 = random_indicies[(len(random_indicies)-1-i) % len(random_indicies)]
        node_photo_id_2 = neighbors_of_selected_nodes[node_key_index_2]
        experiment["edges"].append(",".join("{}".format(n) for n in [node_photo_id_1, node_photo_id_2,0,experiment["false_matching_confidence"],year]))

    node_to_photo = {}
    for line in experiment["nodes"][1:]:
      node_photo_id = line.split(",")[0]
      node = node_photo_id.split(":")[0]
      if node not in node_to_photo:
        node_to_photo[node] = []
      node_to_photo[node].append(node_photo_id.split(":")[1])
    if experiment["find_matches"].lower() == "true":
      for node in node_to_photo:
        for photo in node_to_photo[node][1:]:
          edge = (node + ":" + node_to_photo[node][0], node + ":" + photo)
          experiment["edges"].append(",".join("{}".format(n) for n in [edge[0], edge[1],0,experiment["true_matching_confidence"],"0"]))

    payload["experiments"].append(experiment)
  with open("data/synthetic/experiments.json", "w") as outfile:
    json.dump(payload, outfile, sort_keys=True, indent=2)
if __name__ == "__main__":
  main()