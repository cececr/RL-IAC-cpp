/**
 * \file environment.h
 * \brief Environment
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 9 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <map>
#include <cv.h>
#include "worldmapgraph.h"

class Environment
{
public:
    Environment();
    void init(std::string inputs_path,
                std::vector<std::string> inputs_list,
                std::map<std::string, int> input_region_map = std::map<std::string, int>(),
                WorldMapGraph::GraphStruct graph_struct = WorldMapGraph::GraphStruct());
    void get_inputs(int idx, cv::Mat& rgb_input,cv::Mat& depth_input);
    void get_segmentation_input(int idx, cv::Mat& seg_input);
    int get_region(int current_position);
    int get_inputs_size();
    int get_nb_regions();
    WorldMapGraph get_world_map_graph();
    std::vector<std::vector<int> > get_positions_per_region();
    std::string get_input_name(int current_position);
    std::vector<signed int> get_region_learner_list();
    bool init_world_map(int initial_region);
    cv::Mat draw_world_map(int current_region,
                           std::vector<float> progress = std::vector<float>());

private:
    WorldMapGraph world_map_graph;
    std::map<std::string, int> input_region_map;
    std::vector<std::string> inputs_list;
    std::string inputs_path;
};

#endif // ENVIRONMENT_H
