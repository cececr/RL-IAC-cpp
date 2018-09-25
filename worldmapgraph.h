/**
 * \file worldmapgraph.h
 * \brief WorldMapGraph
 * \author Céline Craye
 * \version 0.1
 * \date 1 / 4 / 2016
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef WORLDMAPGRAPH_H
#define WORLDMAPGRAPH_H
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <map>
#include <cv.h>

class WorldMapGraph
{
public:

    typedef std::vector<std::vector<std::vector<int> > > GraphStruct;
    WorldMapGraph();
    bool init(GraphStruct raw_graph_struct);
    bool init_trivial(std::vector<int> trivial_struct);
    int get_nb_nodes();
    int get_nb_actions();
    int get_next_node(int node, int action);
    int get_weight(int node, int action);
    int get_displacement(int node1, int node2);
    std::vector<signed int> get_region_learner_list();
    std::vector<int> get_reachable_regions(int current_region);

    /* Drawing functions */
    bool init_world_map(int initial_region);
    cv::Mat draw_world_map();
    cv::Point get_node_position(int regionIdx);

    /* best path */
    std::vector<int> find_best_path(int current_region, int new_region);

private:
    bool INIT;
    std::vector<std::vector<int> > displacement_struct;
    std::vector<std::vector<int> > weights_struct;
    std::vector<signed int> learner_list;
    int nb_nodes;
    int nb_actions;

    /* Drawing elements */
    bool DRAWING_INIT;
    std::vector<cv::Point> shifts;
    std::vector<cv::Point> node_position;
    std::vector<int> node_value;
    std::vector<std::vector< cv::Point> > edge_list;
    cv::Point world_map_bounds;

    bool add_to_map(int node, cv::Point point, std::vector<int> & drawn_nodes);

    /* to find best path */
    bool visit_node(std::vector<int> path, std::vector<bool> visited_regions);
    float get_path_dist(std::vector<int> path);
    float min_dist;
    std::vector<int> best_path;
    int new_region;
};

#endif // WORLDMAPGRAPH_H
