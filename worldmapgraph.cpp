/**
 * \file worldmapgraph.cpp
 * \brief WorldMapGraph
 * \author Céline Craye
 * \version 0.1
 * \date 1 / 4 / 2016
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#include "worldmapgraph.h"
#include "common.h"

using namespace std;
using namespace cv;
using namespace Common;

WorldMapGraph::WorldMapGraph()
{
    INIT = false;
    DRAWING_INIT = false;
}

bool WorldMapGraph::init(WorldMapGraph::GraphStruct raw_graph_struct)
{
    assert(!raw_graph_struct.empty());
    assert(!raw_graph_struct[0].empty());
    nb_nodes = raw_graph_struct.size();
    nb_actions = raw_graph_struct[0][0].size();
    for(size_t i = 0 ; i < raw_graph_struct.size() ; i++ )
    {
        displacement_struct.push_back(raw_graph_struct[i][0]);
        weights_struct.push_back(raw_graph_struct[i][1]);
        learner_list.push_back(raw_graph_struct[i][2][0]);
    }
    INIT = true;
    return true;
}
/* Make a fully connected graph */
bool WorldMapGraph::init_trivial(std::vector<int>  trivial_struct)
{

    /* get number of regions */
    nb_nodes = *std::max_element(trivial_struct.begin(), trivial_struct.end())+1;
    nb_actions = trivial_struct.size();
    cout << "nbnodes " <<nb_nodes << " nb_actions " << nb_actions << endl;

    for(int i = 0 ; i < nb_nodes ; i++)
    {
        std::vector<int> displacements;
        std::vector<int> weights;
        for (int j = 0 ; j < nb_actions ; j++)
        {
        /* make each region fully connected with each other */
        displacements.push_back(trivial_struct[j]);
        /* make each weight the same */
        weights.push_back(1);
        }
        /* make a single learner for each */
        learner_list.push_back(0);
        displacement_struct.push_back(displacements);
        weights_struct.push_back(weights);
    }
    INIT = true;
    return true;
}

std::vector<signed int> WorldMapGraph::get_region_learner_list()
{
    return learner_list;
}

std::vector<int> WorldMapGraph::get_reachable_regions(int current_region)
{
    std::vector<int> reachable_list;
    reachable_list.push_back(current_region);
    bool set_has_changed = true;
    int ct = 0;
    while(set_has_changed)
    {
        ct++;
        int s = reachable_list.size();
        for(int i = 0 ;  i < s ; i++)
        {
            reachable_list.insert(reachable_list.end(),
                                  displacement_struct[reachable_list[i]].begin(),
                                  displacement_struct[reachable_list[i]].end());
        }

        sort( reachable_list.begin(), reachable_list.end() );
        vector<int>::iterator it;
        it = std::unique (reachable_list.begin(), reachable_list.end());
        reachable_list.resize( std::distance(reachable_list.begin(),it) );
        if(reachable_list.size() == s)
        {
            set_has_changed = false;
        }
    }
    return reachable_list;
}

int WorldMapGraph::get_nb_nodes()
{
    return nb_nodes;
}

int WorldMapGraph::get_nb_actions()
{
    return nb_actions;
}

int WorldMapGraph::get_next_node(int node, int action)
{
    if(action < nb_actions)
        return displacement_struct[node][action];
    else
        return node;
}

int WorldMapGraph::get_weight(int node, int action)
{
    if(action < nb_actions)
        return weights_struct[node][action];
    else
        return 1;
}

int WorldMapGraph::get_displacement(int node1, int node2)
{
    int displacement = 0;
    bool found_next = false;
    for(int j = 0 ; j < nb_actions ; j++)
    {
        if(displacement_struct[node1][j] == node2)
        {
             displacement = weights_struct[node1][j]
                       + FLT_EPSILON*(float)rand()/RAND_MAX;
             found_next =  true;
             break;
        }
    }
    if(!found_next)
        cerr << "displacement from "
             <<node1 << " to "
             << node2 << "is not allowed" << endl;
    return displacement;
}

bool WorldMapGraph::init_world_map(int initial_region)
{
    if(!INIT)
        return false;

    /* init action shifts */
    int step = 10;
    shifts.push_back(cv::Point(0,-step));
    shifts.push_back(cv::Point(0,step));
    shifts.push_back(cv::Point(-step,0));
    shifts.push_back(cv::Point(step,0));
    shifts.push_back(cv::Point(0,0));

    /* init drawn nodes */
    vector<int> drawn_nodes;
    for(int i = 0 ; i < nb_nodes; i++)
    {
        node_position.push_back(cv::Point());
    }
    for(size_t i = 0 ; i < nb_nodes ; i++)
    {
        if(nb_actions != shifts.size())
        {
            cerr << "Provided graph map cannot be displayed" << endl;
            return false;
        }
        drawn_nodes.push_back(false);
    }
    cout << nb_nodes << endl;
    cv::Point pt(0,0);
    add_to_map(initial_region,pt, drawn_nodes);
    /*shift all points for better display*/
    cv::Point min_loc = pt;
    cv::Point max_loc = pt;
    for(size_t i = 0 ; i < node_position.size() ; i++)
    {
        if(min_loc.x > node_position[i].x)
            min_loc.x = node_position[i].x;
        if(min_loc.y > node_position[i].y)
            min_loc.y = node_position[i].y;
        if(max_loc.x < node_position[i].x)
            max_loc.x = node_position[i].x;
        if(max_loc.y < node_position[i].y)
            max_loc.y = node_position[i].y;
    }
    cv::Point draw_shift(step - min_loc.x,step - min_loc.y);
    world_map_bounds = cv::Point(step+max_loc.x + draw_shift.x,
                                 step+max_loc.y + draw_shift.y);
    for(size_t i = 0 ; i < node_position.size() ; i++)
    {
        node_position[i].x +=  draw_shift.x;
        node_position[i].y +=  draw_shift.y;
    }
    for(size_t i = 0 ; i < edge_list.size() ; i++)
    {
        edge_list[i][0].x +=  draw_shift.x;
        edge_list[i][0].y +=  draw_shift.y;
        edge_list[i][1].x +=  draw_shift.x;
        edge_list[i][1].y +=  draw_shift.y;
    }

    /* Max map size is 1000*1000 */
    float max_map_size = 1000;
    float mulFactor = min(max_map_size/(float)(world_map_bounds.x-2*step),
                          max_map_size/(float)(world_map_bounds.y-2*step));
    world_map_bounds.y = mulFactor*world_map_bounds.y;
    world_map_bounds.x = mulFactor*world_map_bounds.x;

    /* multiply edge list by mulFactor */
    for(size_t i = 0 ; i < node_position.size() ; i++)
    {
        node_position[i].x *=  mulFactor;
        node_position[i].y *=  mulFactor;
    }
    for(size_t i = 0 ; i < edge_list.size() ; i++)
    {
        edge_list[i][0].x *=  mulFactor;
        edge_list[i][0].y *=  mulFactor;
        edge_list[i][1].x *=  mulFactor;
        edge_list[i][1].y *=  mulFactor;
    }

    DRAWING_INIT = true;
    return true;
}

cv::Mat WorldMapGraph::draw_world_map()
{
    if(!DRAWING_INIT)
        return cv::Mat();


    /* Create map */
    cv::Mat world_map = cv::Mat::zeros(world_map_bounds.y,
                                       world_map_bounds.x,CV_8UC3);
    Scalar white = Scalar(255,255,255);



    /* Draw edges */
    for(size_t i = 0 ; i < edge_list.size() ; i++)
    {
        cv::line(world_map, edge_list[i][0], edge_list[i][1],white,1);
    }

    /* Draw nodes */
    for(size_t i = 0 ; i < node_position.size() ; i++)
    {
        cv::Point textpoint = node_position[i];
        textpoint.y = textpoint.y - 10;
        cv::putText(world_map, to_string(i),
                    textpoint, cv::FONT_HERSHEY_SIMPLEX,0.5,white);
        cv::circle(world_map, node_position[i],5 ,white,CV_FILLED);
    }

    return world_map;
}

cv::Point WorldMapGraph::get_node_position(int regionIdx)
{
    cv::Point node_pos;
    (regionIdx < 0 || regionIdx >= node_position.size()) ?
        node_pos = Point(0,0) : node_pos = node_position[regionIdx];
    return node_pos;
}

bool WorldMapGraph::add_to_map(int node, cv::Point point, vector<int> &drawn_nodes)
{
    if(drawn_nodes[node])
        return false;

    node_position[node] = point;
    drawn_nodes[node] = true;

    for(size_t i = 0 ; i < displacement_struct[node].size() ; i++)
    {
        int neighbor = displacement_struct[node][i];
        int weight = weights_struct[node][i];
        if(neighbor != node)
        {
            cv::Point neighbor_loc(point.x + weight * shifts[i].x ,
                                   point.y + weight * shifts[i].y);
            vector<cv::Point> edge;
            edge.push_back(point);
            edge.push_back(neighbor_loc);
            edge_list.push_back(edge);
            add_to_map(neighbor,neighbor_loc, drawn_nodes);
        }
    }
    return true;
}

vector<int> WorldMapGraph::find_best_path(int current_region, int new_region)
{
    vector<bool> visited_regions;
    for(int i = 0 ; i < nb_nodes ; i++)
        visited_regions.push_back(false);
    vector<int> path;
    path.push_back(current_region);
    visited_regions[current_region] = true;
    min_dist = -1;
    best_path.clear();
    this->new_region = new_region;
    visit_node(path, visited_regions);
    return best_path;
}

bool WorldMapGraph::visit_node(vector<int> path, vector<bool> visited_regions)
{
    int node = path[path.size()-1];
    if(node == new_region)
    {
        float dist = get_path_dist(path);
        if(best_path.empty() || dist < min_dist)
        {
            best_path = path;
            min_dist = dist;
        }
    }
    else
    {
        for(int i = 0 ; i < nb_actions ; i++)
        {
            int neighbor = displacement_struct[node][i];
            if(visited_regions[neighbor] == true)
                continue;
            vector<int> new_path = path;
            new_path.push_back(neighbor);
            if(min_dist > 0 && get_path_dist(new_path) >= min_dist)
                continue;
            visited_regions[neighbor] = true;
            visit_node(new_path, visited_regions);
        }
    }
    path.pop_back();
    visited_regions[node] = false;
    return true;
}

float WorldMapGraph::get_path_dist(std::vector<int> path)
{
    float path_dist = 0;
    if(path.size() >= 1)
    {
        for(size_t i = 0 ;  i < path.size()-1 ; i++)
        {
            int current_region = path[i];
            int next_region = path[i+1];
            for(int j = 0 ; j < nb_actions ; j++)
            {
                if(displacement_struct[current_region][j] == next_region)
                {
                    path_dist += weights_struct[current_region][j]
                               + FLT_EPSILON*(float)rand()/RAND_MAX;
                }
            }
        }
    }
    return path_dist;
}
