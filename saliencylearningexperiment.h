/**
 * \file saliencylearningexperiment.h
 * \brief SaliencyLearningExperiment
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 9 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef SALIENCYLEARNINGEXPERIMENT_H
#define SALIENCYLEARNINGEXPERIMENT_H

#include <iostream>
#include <cv.h>

#include "RL_IAC/RegionsM.h"
#include "RL_IAC/ActionSelectionM.h"
#include "environment.h"
#include "foveal_segmentation/pt_cld_segmentation.h"
#include "foveal_segmentation/floortracker.h"
#include "feature_extractor/FeatureExtractor.h"
#include "learning_module/dataformater.h"
#include "bottom_up/bottomupsaliencymethod.h"

#include "SaliencyFileParser.h"
#include "experimentevaluation.h"
#include "printdebug.h"

typedef pcl::PointXYZRGB PointT;

class SaliencyLearningExperiment
{
public:
    SaliencyLearningExperiment(std::string param_file = "");
    ~SaliencyLearningExperiment();
    void run(); // the main loop
    void run_bottom_up();
    void run_segmentation_only();
    void run_offline_learning();
    void run_offline_saliency();

private:
    /* Main loop functions */
    bool take_action(int i);
    bool get_input();
    bool extract_features();
    bool segment_objects();
    bool feed_learners();
    bool train_learners();
    bool evaluate();
    bool get_saliency_map();
    bool display();
    bool save_model();
    bool load_model();
    bool save_saliency_map();

    /* initialization function */
    bool init_environment();
    bool init_experiment();
    bool init_feature_extractor();
    bool init_segmenter();
    bool init_learner();
    bool init_action_selection();
    bool init_evaluation();
    bool init_bottom_up();
    bool init_missing_params();
    void test_num_key(std::string key, float default_value);
    void test_string_key(std::string key, string default_value);

    /* cv buffers */
    cv::Mat rgb_input;
    cv::Mat depth_input;
    cv::Mat segmentation_mask;
    cv::Mat ground_truth;
    cv::Mat saliency_map;
    cv::Mat fused_map;
    std::vector<cv::Mat> feature_map;
    std::vector<float> region_states;
    bool learn;

    /* modules */
    Environment environment;
    ActionSelectionM action_selector;
    FeatureExtractor* feature_extractor;
    PtCldSegmentation<PointT> object_segmenter;
    RegionsM regions_learner;
    ExperimentEvaluation experiment_eval;
    FloorTracker<PointT> floor_tracker;
    BottomUpSaliencyMethod* bottom_up_extractor;


    /* Parameters */
    std::string params_filename;
    SaliencyFileParser::Param_Struct global_params;
    int initial_position;
    int nb_steps;

    /*Robot state*/
    int time;
    int current_position;

};

#endif // SALIENCYLEARNINGEXPERIMENT_H
