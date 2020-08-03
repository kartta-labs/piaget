
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

  def haversine_distance(self, point1, point2):
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
      
      node_to_neighbors[year] = {}
      edges = map.get_all_edges()[year]
      for edge in edges:
        node_to_neighbors[year].setdefault(edge[0], set()).add(edge[1])
        node_to_neighbors[year].setdefault(edge[1], set()).add(edge[0])

      nodes = map.get_all_nodes()[year]
      node_keys = list(nodes.keys())
      random.seed(experiment["randomness_seed"])
      random_indicies = [random.randint(0, len(node_keys)) for i in range(0, number_of_unique_photos)]
      random_indicies_of_seeds = [random.randint(0, len(random_indicies)) for i in range(0, int(number_of_unique_photos*ratio_of_seeds))]
      selected_nodes = set()
      for i in range(len(random_indicies)):
        node_key_index = random_indicies[i]
        node = node_keys[node_key_index]
        selected_nodes.add(node)
        seed = i in random_indicies_of_seeds
        experiment["nodes"].append(",".join("{}".format(n) for n in [node, nodes[node][1], nodes[node][0],seed,seed,1e-8,0,1e-8,year]))

      edges = map.get_all_edges()[year]
      for node in selected_nodes:
        for neighbor in node_to_neighbors[year][node]:
          edge = (node,neighbor)
          if (node,neighbor) not in edges:
            edge = (neighbor,node)
          experiment["edges"].append(",".join("{}".format(n) for n in [edge[0], edge[1],edges[edge],0.1,year]))
          experiment["nodes"].append(",".join("{}".format(n) for n in [neighbor, nodes[neighbor][1], nodes[neighbor][0],False,False,1e-8,0,1e-8,year]))
    payload["experiments"].append(experiment)
  with open("data/synthetic/experiments.json", "w") as outfile:
    json.dump(payload, outfile, sort_keys=True, indent=2)
if __name__ == "__main__":
  main()