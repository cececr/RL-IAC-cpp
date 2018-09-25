/**
 * \file environment.cpp
 * \brief Environment
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 9 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */
#include <highgui.h>

#include "environment.h"
#include "printdebug.h"

using namespace std;


Environment::Environment()
{
}

void Environment::init(std::string inputs_path,
                         std::vector<std::string> inputs_list,
                         std::map<std::string, int> input_region_map,
                         WorldMapGraph::GraphStruct graph_struct)
{

    this->inputs_list = inputs_list;
    this->inputs_path = inputs_path;

    /* If input region map is missing generate trivial one */
    if(input_region_map.size() == 0)
    {
        for(size_t i = 0 ; i < inputs_list.size() ; i++)
        {
            input_region_map[inputs_list[i]] = 0;
        }
    }
    this->input_region_map = input_region_map;

    /* if input list is empty, generate from the map */
    if(inputs_list.empty())
    {
        //loop through map
        typedef std::map<std::string, int>::iterator it_type;
        for(it_type iterator = input_region_map.begin();
            iterator != input_region_map.end(); iterator++) {
            this->inputs_list.push_back(iterator->first);
        }
    }
    else
    {
        /* remove inputs that are not in the map */
        std::vector<std::string> clean_inputs_list;
        map<std::string, int>::iterator it;
        for(size_t i = 0 ; i < this->inputs_list.size() ; i++)
        {
            it = input_region_map.find(this->inputs_list[i]);
            if( it != input_region_map.end())
                clean_inputs_list.push_back(this->inputs_list[i]);
        }
        this->inputs_list = clean_inputs_list;
        cout << "inputs_list " << clean_inputs_list.size() << endl;
    }

    /* if graph struct is missing, generate trivial one */
    if(graph_struct.empty())
    {
        // Get number of regions
        vector<int> regions_list ;
        typedef std::map<std::string, int>::iterator it_type;
        for(it_type iterator = input_region_map.begin();
            iterator != input_region_map.end(); iterator++)
        {
            regions_list.push_back(iterator->second);
        }
        std::unique (regions_list.begin(), regions_list.end());

        this->world_map_graph.init_trivial(regions_list);
    }
    else
    {
        this->world_map_graph.init(graph_struct);
    }
}

void Environment::get_inputs(int idx,cv::Mat &rgb_input,
                             cv::Mat &depth_input)
{
    if(idx >= 0 && idx < inputs_list.size() )
    {
        /* Load RGB input image */
        std::string img = inputs_path + "/" + inputs_list[idx];
        rgb_input = cv::imread(img);

        /* Load depth map */
        std::string dph = img;
        dph.replace(img.length()-4,0,"_depth");
        depth_input = cv::imread(dph,CV_LOAD_IMAGE_ANYDEPTH);
    }
    else
    {
        /* Create default matrices */
//        rgb_input = cv::Mat::zeros(480,640,CV_8UC3);
//        depth_input = cv::Mat::zeros(480,640,CV_16U);
    }
}

void Environment::get_segmentation_input(int idx, cv::Mat &seg_input)
{
    if(idx >= 0 && idx < inputs_list.size() )
    {
    /* Load segmentation input image */
    std::string img = inputs_path + "/" + inputs_list[idx];
    img.replace(img.length()-4,0,"_GT");
    seg_input = cv::imread(img);
    }
    else
    {
        /* Create default matrices */
        seg_input = cv::Mat::zeros(480,640,CV_8UC3);
    }

    if(seg_input.channels() == 3)
        cvtColor(seg_input,seg_input,CV_RGB2GRAY);

}

/**
 * @brief Environment::get_region get region index from current position index
 * @param current_position the current position index
 * @return the region index corresponding to current position
 */
int Environment::get_region(int current_position)
{
    if(abs(current_position) >= (int)inputs_list.size())
        return -1;
    if(current_position < 0)
        return - current_position;

    return input_region_map[inputs_list[current_position]];
}

int Environment::get_inputs_size()
{
    return inputs_list.size();
}

int Environment::get_nb_regions()
{
    return world_map_graph.get_nb_nodes();
}

WorldMapGraph Environment::get_world_map_graph()
{
    return world_map_graph;
}

/**
 * @brief Environment::get_positions_per_region
 * @return list of all image indices for each region
 */
std::vector<std::vector<int> > Environment::get_positions_per_region()
{
    std::vector<std::vector<int> > positions_per_region;
    /* Initialize vector for each region */
    for(int i = 0 ; i < get_nb_regions() ; i++) // TODO : function calls world map graph before it was init
    {
        std::vector<int> region_vect;
        positions_per_region.push_back(region_vect);
    }
    for(size_t i = 0 ; i < inputs_list.size() ; i++)
    {
        int region = input_region_map[inputs_list[i]];
        positions_per_region[region].push_back(i);
    }
    return positions_per_region;
}

std::string Environment::get_input_name(int current_position)
{
    return inputs_list[current_position];
}

std::vector<signed int> Environment::get_region_learner_list()
{
    return world_map_graph.get_region_learner_list();
}

bool Environment::init_world_map(int initial_region)
{
    return world_map_graph.init_world_map(initial_region);
}

cv::Mat Environment::draw_world_map(int current_region, vector<float> progress)
{
    /* Draw graph  */
    cv::Mat world_map = world_map_graph.draw_world_map();

    /* Draw robot */
    if(!world_map.empty())
    {
        cv::circle(world_map,
                   world_map_graph.get_node_position(current_region),
                   8 ,cv::Scalar(255,0,0) , 2);
    }

    /* Draw progress state */
    if(!world_map.empty() && !progress.empty())
    {
        double min, max;
        cv::minMaxLoc(cv::Mat(progress), &min,&max);
        if(min == max)
            min = max-1;
//        min = 0; max = 10;
        for(size_t i = 0 ; i < progress.size() ; i++)
        {
            float h = 50*(progress[i]-min)/(max-min);
            float s = 255, v = 255;
            cv::Mat pixvalue = (cv::Mat_<uchar>(3,1)<<h,s,v);
            pixvalue = pixvalue.reshape(3,1);
            cv::cvtColor(pixvalue, pixvalue, CV_HSV2BGR);
            pixvalue = pixvalue.reshape(1,3);
            cv::Scalar color = cv::Scalar(pixvalue.at<uchar>(0),
                                          pixvalue.at<uchar>(1),
                                          pixvalue.at<uchar>(2));
            cv::circle(world_map,
                       world_map_graph.get_node_position(i),
                       5 ,color , CV_FILLED);
        }
    }

    return world_map;
}
