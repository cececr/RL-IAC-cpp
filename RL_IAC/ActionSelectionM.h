/**
 * \file ActionSelectionM.h
 * \brief ActionSelectionM
 * \author Céline Craye
 * \version 0.1
 * \date 7 / 23 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef ACTIONSELECTIONM_H
#define ACTIONSELECTIONM_H

#include "RegionsM.h"
#include "worldmapgraph.h"
#include <limits>

struct ExplorationParams
{
    float IAC_GREEDY_EPS = 0.3f;
    int   IAC_DISTRIBUTION_POWER = 3;
    float RLIAC_RANDOM_POS_EPS = 0.1f;//0.1
    int   RLIAC_AVAILABLE_TIME_PER_EPISODE = 100;
    int   RLIAC_NB_EPISODES = 1000;
    float RLIAC_DISCOUNT_FACTOR = 0.3f;
    float RLIAC_GREEDY_EPS = 0.5f;
    float RLIAC_RAND_SELECT_EPS = 0.1f;
};

class ActionSelectionM
{
public:
    static const int SELECT_CHRONO = 0;
    static const int SELECT_RANDOM = 1;
    static const int SELECT_IAC = 2;
    static const int SELECT_RLIAC = 3;
    static const int SELECT_Schmidhuber = 4;

    static const int LEARN = INT_MAX;
    static const int LEARN_TIME = 1;
    static const int USELESS_ACTION_REWARD = -100;
    static const int RAND_NORM = 10000;

    int minval;
    int maxval;

    ActionSelectionM();
    bool init(int selection_type,
              WorldMapGraph world_map_graph,
              vector<vector<int> > positions_per_region,
              bool learn_at_each_step = true,
              int nb_learn_per_stay = 2,
              int learn_move_ratio = 1,
              ExplorationParams params = ExplorationParams());
    void select_position(int current_region,
                         vector<float> region_progress_states,
                         int *selected_position,
                         int *time_to_pos);
    int get_selection_type();

private:
    int selection_type;
    WorldMapGraph world_map_graph;
    vector<vector<int> > positions_per_region;
    vector<int> path_plan;
    vector<int> time_plan;
    cv::Mat reward_table;
    int learn_val;
    int learn_time;
    bool learn_at_each_step ;
    int nb_learn_per_stay ;
    int learn_move_ratio ;
    ExplorationParams params;

    // for Schmidhuber
    cv::Mat Q_schmid;
    cv::Mat alpha_schmid;


    void select_chrono();
    void select_target_position(int current_region,
                                vector<float> region_progress_states,
                                int selection_type);
    int select_random(int current_region);
    int select_IAC(vector<float> region_progress_states, int current_region);
    void select_RLIAC(int current_region, vector<float> region_progress_states);
    void select_Schmidhuber(int current_region, vector<float> region_progress_states);
    void get_useless_actions();
    void init_Q_reward_alpha(cv::Mat &Q, cv::Mat &R, cv::Mat &alpha,
                             vector<float> region_progress_states);
    void update_reward(cv::Mat &R,vector<float> region_progress_states);
    int select_action(int state, cv::Mat & Q, float epsilon = 0);
    void get_RL_path_plan(int current_region, cv::Mat &Q);
    void get_Schmidhuber_path_plan(int current_region, int action);

    int select_pos_in_region(int regionIdx);
    void add_learn_to_plan();
    void add_displacement_to_plan(int region_before, int region_after);

    float rand_norm();

};

#endif // ACTIONSELECTIONM_H
