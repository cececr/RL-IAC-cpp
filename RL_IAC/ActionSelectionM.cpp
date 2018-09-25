/**
 * \file ActionSelectionM.cpp
 * \brief ActionSelectionM
 * \author Céline Craye
 * \version 0.1
 * \date 7 / 23 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#include "ActionSelectionM.h"
#include "printdebug.h"


ActionSelectionM::ActionSelectionM()
{
    this->selection_type = SELECT_CHRONO;
    learn_val = LEARN;
    learn_time = LEARN_TIME;
}

/**
 * @brief ActionSelectionM::init
 * @param selection_type
 * @param world_map_graph
 * @param positions_per_region a vector where each element is a list of frames indices
 *        representing the frames in a given region
 * @return
 */
bool ActionSelectionM::init(int selection_type,
                            WorldMapGraph world_map_graph,
                            vector<vector<int> > positions_per_region,
                            bool learn_at_each_step,
                            int nb_learn_per_stay,
                            int learn_move_ratio, ExplorationParams params)
{
    this->selection_type = selection_type;
    this->world_map_graph = world_map_graph;
    this->positions_per_region = positions_per_region;
    if(selection_type == SELECT_RLIAC || selection_type == SELECT_Schmidhuber)
    {// IN RL_IAC, this should be always true (otherwise never learns !)
        this->learn_at_each_step = true;
    }
    else
    {
        this->learn_at_each_step = learn_at_each_step;
    }
    this->nb_learn_per_stay = nb_learn_per_stay;
    this->learn_move_ratio = learn_move_ratio;
    this->params = params;
    if(selection_type == SELECT_RLIAC || selection_type == SELECT_Schmidhuber)
        get_useless_actions();
    return true;
}

void ActionSelectionM::select_position(int current_region,
                                      vector<float> region_progress_states,
                                      int* selected_position, int* time_to_pos)
{
    *selected_position = -1;
    *time_to_pos = 0;

    if(path_plan.empty())
    {
        switch (selection_type) {
        case SELECT_CHRONO:
            cout << "SELECT_CHRONO"<< endl;
            select_chrono();
            break;
        case SELECT_RANDOM:
            cout << "SELECT_RANDOM"<< endl;
            select_target_position(current_region,region_progress_states,selection_type);
            break;
        case SELECT_IAC:
            cout << "SELECT_IAC"<< endl;
            select_target_position(current_region,region_progress_states,selection_type);
            break;
        case SELECT_RLIAC:
            cout << "SELECT_RLIAC"<< endl;
            select_RLIAC(current_region, region_progress_states);
            break;
        case SELECT_Schmidhuber:
            cout << "SELECT_Schmidhuber"<< endl;
            select_Schmidhuber(current_region, region_progress_states);
            break;
        default:
            break;
        }
    }
    *selected_position = path_plan[0];
    path_plan.erase(path_plan.begin());
    *time_to_pos = time_plan[0];
    time_plan.erase(time_plan.begin());
}

int ActionSelectionM::get_selection_type()
{
    return selection_type;
}

void ActionSelectionM::select_chrono()
{
    /* If empty, fill path plan */
    for(size_t i = 0 ; i < positions_per_region.size() ; i++)
    {
        for(size_t j = 0 ; j < positions_per_region[i].size() ; j++)
        {
            path_plan.push_back(positions_per_region[i][j]);
            time_plan.push_back(learn_time);
            add_learn_to_plan();
        }
    }
}

void ActionSelectionM::select_target_position(int current_region,
                                             vector<float> region_progress_states,
                                             int selection_type)
{
    /* Select new region */
    vector<int> best_region_path;
    int new_region = -1;
    if(selection_type == SELECT_RANDOM)
    {
        new_region = select_random(current_region);
        best_region_path.push_back(new_region);
    }
    else if (selection_type == SELECT_IAC)
    {
        new_region = select_IAC(region_progress_states, current_region);
        best_region_path.push_back(new_region);
    }
    else
    {
    /* find min distance between the two */
    best_region_path = world_map_graph.find_best_path(current_region, new_region);
    }
    /* add random positions in each visited region */
    size_t startIdx = 0;
    best_region_path.size() > 1 ? startIdx = 1 : startIdx = 0;
    int region_before = best_region_path[0];
    for(size_t i = startIdx ; i < best_region_path.size() ; i++)
    {
        int region_after = best_region_path[i];
        add_displacement_to_plan(region_before,region_after);
        region_before = region_after;
        if(learn_at_each_step)
        {
            add_learn_to_plan();
        }
    }

    /* plan to learn once at destination */
    int final_region = best_region_path.back();
    for(int i = 0 ; i < nb_learn_per_stay ; i++)
    {
        add_displacement_to_plan(final_region,final_region);
        add_learn_to_plan();
    }
}

int ActionSelectionM::select_random(int current_region)
{
    cout << "select random" << endl;
    /* Make sure region is reachable in the nav graph */
    vector<int> reachable = world_map_graph.get_reachable_regions(current_region);
    int targer_region = reachable[rand() % reachable.size()];
    return targer_region;
}

int ActionSelectionM::select_IAC(vector<float> region_progress_states, int current_region)
{
    assert(!region_progress_states.empty());
    assert((int)region_progress_states.size()
           == world_map_graph.get_nb_nodes());

    /* select randomly once in a while */
    if(rand_norm() < params.IAC_GREEDY_EPS)
    {
        return select_random(current_region);
    }


    /* make a probabilistic table based on progress value */
    vector<int> reachable_regions = world_map_graph.get_reachable_regions(current_region);
    vector<float> region_progress_weight(region_progress_states.size(),0);
    for(int i = 0 ; i < reachable_regions.size() ; i++)
    {
        region_progress_weight[reachable_regions[i]] = 1;
    }
    float offset = FLT_EPSILON;
    vector<float> cumulated_progress;
    cumulated_progress.push_back(pow(region_progress_states[0],3) + offset);
    for(size_t i = 1 ; i < region_progress_states.size() ; i++)
    {
        cumulated_progress.push_back(cumulated_progress[i-1]
                                     + pow(region_progress_weight[i]*region_progress_states[i],
                                            params.IAC_DISTRIBUTION_POWER)
                                     + offset);
    }
    /* select randomly based on probabilistic table */
    float rand_value = rand_norm()* cumulated_progress[cumulated_progress.size()-1];
    for(size_t i = 0 ; i < cumulated_progress.size() ; i++)
    {
        if(rand_value < cumulated_progress[i])
        {
            return i;
        }
    }
    return 0;
}

std::string action_string(int action)
{
    switch (action) {
    case 0:
        return "up";
    case 1:
        return "down";
    case 2:
        return "left";
    case 3:
        return "right";
    case 4:
        return "stay";
    default:
        return "learn";
    }
}

void ActionSelectionM::select_Schmidhuber(int current_region, vector<float> region_progress_states)
{

    /* Set little progresses to 0 */
    double minProg, maxProg;
    cv::minMaxLoc(cv::Mat(region_progress_states),&minProg,&maxProg);
    for(size_t i = 0 ; i < region_progress_states.size() ; i++)
    {
        if(region_progress_states[i]< maxProg/2)
            region_progress_states[i] = 0;
    }
    cout << "train Q matrix" << endl;
    /* Train Q matrix */
    int N = params.RLIAC_AVAILABLE_TIME_PER_EPISODE;
    int nb_episodes = params.RLIAC_NB_EPISODES;
    float gamma = params.RLIAC_DISCOUNT_FACTOR;
    float epsilon = params.RLIAC_GREEDY_EPS;

    cv::Mat R;
    if(Q_schmid.empty())
    {
        init_Q_reward_alpha(Q_schmid, R, alpha_schmid, region_progress_states);
    }
    else
    {
        update_reward(R, region_progress_states);
    }
    double maxQ;

    int state = current_region;
    // select action for the current state
    int action = select_action(state,Q_schmid, epsilon);
    // get reward
    float reward = R.at<float>(state, action);
    // consider going to the next state
    int next_state = world_map_graph.get_next_node(state, action);
    // get max Q for this next state
    cv::Mat Qline = Q_schmid.row(next_state);
    cv::minMaxLoc(Qline, NULL, &maxQ);
    // update Q(state, action)
    Q_schmid.at<float>(state, action) = Q_schmid.at<float>(state, action)
                              + 1/alpha_schmid.at<float>(state,action)
                              * (reward + gamma*maxQ
                                - Q_schmid.at<float>(state, action));
    alpha_schmid.at<float>(state,action) ++;

    /* select randomly once in a while */
    if(rand_norm() < params.RLIAC_RANDOM_POS_EPS)
    {
        cout << "select random" << endl;
        select_target_position(current_region,
                               region_progress_states,
                               SELECT_RANDOM);
        return;
    }


    /* Get best action and next position */
    cout << "get path plan " << endl;
    get_Schmidhuber_path_plan(current_region, action);

}

void ActionSelectionM::select_RLIAC(int current_region, vector<float> region_progress_states)
{
    /* select randomly once in a while */
    if(rand_norm() < params.RLIAC_RANDOM_POS_EPS)
    {
        cout << "select random" << endl;
        select_target_position(current_region,
                               region_progress_states,
                               SELECT_RANDOM);
        return;
    }

    /* Set little progresses to 0 */
    double minProg, maxProg;
    cv::minMaxLoc(cv::Mat(region_progress_states),&minProg,&maxProg);
    for(size_t i = 0 ; i < region_progress_states.size() ; i++)
    {
        if(region_progress_states[i]< maxProg/2)
            region_progress_states[i] = 0;
    }
    cout << "train Q matrix" << endl;
    /* Train Q matrix */
    int N = params.RLIAC_AVAILABLE_TIME_PER_EPISODE;
    int nb_episodes = params.RLIAC_NB_EPISODES;
    float gamma = params.RLIAC_DISCOUNT_FACTOR;
    float epsilon = params.RLIAC_GREEDY_EPS;
    cv::Mat Q , R, alpha;
    init_Q_reward_alpha(Q, R, alpha, region_progress_states);
    double maxQ;
    for(int i = 0 ; i < nb_episodes ; i++)
    {
        int state = current_region;
        int time = 0;
        while(time < N)
        {
            // select action for the current state
            int action = select_action(state,Q, epsilon);
            // get reward
            float reward = R.at<float>(state, action);
            // consider going to the next state
            int next_state = world_map_graph.get_next_node(state, action);
            int action_time = world_map_graph.get_weight(state, action);
            // get max Q for this next state
            cv::Mat Qline = Q.row(next_state);
            cv::minMaxLoc(Qline, NULL, &maxQ);
            // update Q(state, action)
            Q.at<float>(state, action) = Q.at<float>(state, action)
                                      + 1/alpha.at<float>(state,action)
                                      * (reward + gamma*maxQ
                                        - Q.at<float>(state, action));
            alpha.at<float>(state,action) ++;
            // update state and time
            state = next_state;
            time += action_time;
        }
    }
    /* Get best action and next position */
    cout << "get path plan " << endl;
    get_RL_path_plan(current_region, Q);

}

void ActionSelectionM::get_useless_actions()
{
    reward_table = cv::Mat::zeros(world_map_graph.get_nb_nodes(),
                              world_map_graph.get_nb_actions()+1,
                              CV_32F);
    for(int i = 0 ; i < reward_table.rows ; i++)
    {
        for(int j = 0 ; j < reward_table.cols-1 ; j++)
        {
            if(world_map_graph.get_next_node(i,j) == i)
            {
                reward_table.at<float>(i,j) = USELESS_ACTION_REWARD;
            }
            else
            {
                reward_table.at<float>(i,j) = 0;
            }
        }
        // NEW VERSION = LEARN IS USELESS !!
        reward_table.at<float>(i,reward_table.cols-1) = USELESS_ACTION_REWARD;
    }
}

//////////////
/// OLD VERSION
//void ActionSelectionM::init_Q_reward_alpha(cv::Mat &Q,
//                                           cv::Mat &R,
//                                           cv::Mat &alpha,
//                                           vector<float> region_progress_states)
//{
//    Q = cv::Mat::zeros(world_map_graph.get_nb_nodes(),
//                        world_map_graph.get_nb_actions()+1,
//                        CV_32F);
//    alpha = cv::Mat::ones(Q.size(), Q.type());
//    R = reward_table.clone();
//    cv::Mat progress_mat = cv::Mat(region_progress_states);
//    progress_mat.copyTo(R(cv::Rect(R.cols-1,0,progress_mat.cols, progress_mat.rows)));
//}

void ActionSelectionM::init_Q_reward_alpha(cv::Mat &Q,
                                           cv::Mat &R,
                                           cv::Mat &alpha,
                                           vector<float> region_progress_states)
{
    Q = cv::Mat::zeros(world_map_graph.get_nb_nodes(),
                        world_map_graph.get_nb_actions()+1,
                        CV_32F);
    cv::randu(Q,0 ,0.001);
    alpha = cv::Mat::ones(Q.size(), Q.type());
    R = reward_table.clone();
    cv::Mat progress_mat = cv::Mat(region_progress_states);
    /* TODO: Weight progress mat with the size of the region */
    for(int i = 0 ; i < R.cols-1 ; i++)
    {
        progress_mat.copyTo(R(cv::Rect(i,0,progress_mat.cols, progress_mat.rows)));
    }
}

void ActionSelectionM::update_reward(cv::Mat &R, vector<float> region_progress_states)
{
    R = reward_table.clone();
    cv::Mat progress_mat = cv::Mat(region_progress_states);
    /* TODO: Weight progress mat with the size of the region */
    for(int i = 0 ; i < R.cols-1 ; i++)
    {
        progress_mat.copyTo(R(cv::Rect(i,0,progress_mat.cols, progress_mat.rows)));
    }
}


int ActionSelectionM::select_action(int state, cv::Mat &Q, float epsilon)
{
    /* select randomly based on epsilon */
    cv::Mat useless_line = reward_table.row(state);
    if(rand_norm() < epsilon)
    {
        int action;
        do
        {
            action = rand() % Q.cols;
        }
        while(useless_line.at<float>(action)== USELESS_ACTION_REWARD);
        return action;
    }
    /* select based on Q matrix */
    cv::Mat Qline = Q.row(state).clone();
    cv::Mat randline = cv::Mat(Qline.size(), Qline.type());
    cv::randu(randline,0,10*FLT_EPSILON);
    Qline = Qline + randline+useless_line;
    double min, max;
    cv::Point minL, maxL;
    cv::minMaxLoc(Qline, &min, &max, &minL, &maxL);
    return maxL.x;
}

void ActionSelectionM::get_Schmidhuber_path_plan(int current_region, int action)
{
    int nb_action = world_map_graph.get_nb_actions();
    cout << "best action is " << action_string(action) << endl;
    int next_region = world_map_graph.get_next_node(current_region, action);

    if(action >= nb_action) // then learn
    {
        for(int i = 0 ; i < nb_learn_per_stay ; i++)
        {
            add_displacement_to_plan(current_region, current_region);
            add_learn_to_plan();
        }
    }
    else // then move
    {
        add_displacement_to_plan(current_region, next_region);
        if(learn_at_each_step)
            add_learn_to_plan();
    }
}

void ActionSelectionM::get_RL_path_plan(int current_region, cv::Mat & Q)
{
    int nb_action = world_map_graph.get_nb_actions();
    int action = select_action(current_region,Q,params.RLIAC_RAND_SELECT_EPS);

    cout << "best action is " << action_string(action) << endl;
    int next_region = world_map_graph.get_next_node(current_region, action);

    if(action >= nb_action) // then learn
    {
        for(int i = 0 ; i < nb_learn_per_stay ; i++)
        {
            add_displacement_to_plan(current_region, current_region);
            add_learn_to_plan();
        }
    }
    else // then move
    {
        add_displacement_to_plan(current_region, next_region);
        if(learn_at_each_step)
            add_learn_to_plan();
    }
}

int ActionSelectionM::select_pos_in_region(int regionIdx)
{
    if(positions_per_region[regionIdx].empty())
        return -regionIdx;

    int posIdx = rand() % positions_per_region[regionIdx].size();
//    while( ( positions_per_region[regionIdx][posIdx] < minval)
//           || ( positions_per_region[regionIdx][posIdx] > maxval))
//    {
//           posIdx = rand() % positions_per_region[regionIdx].size();
//    }

    return positions_per_region[regionIdx][posIdx];
}

void ActionSelectionM::add_learn_to_plan()
{
    path_plan.push_back(learn_val);
    time_plan.push_back(learn_time);
}

void ActionSelectionM::add_displacement_to_plan(int region_before, int region_after)
{
    int displacement_time = learn_move_ratio
                          * world_map_graph.get_displacement(region_before,region_after);
    time_plan.push_back(displacement_time);
    path_plan.push_back(select_pos_in_region(region_after));
}

float ActionSelectionM::rand_norm()
{

    float randval =  (double)(rand()%RAND_NORM)/(double)RAND_NORM;
//    cout <<"rand val " << randval << endl;
    return randval;
}
