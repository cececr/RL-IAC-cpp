/**
 * \file experimentevaluation.h
 * \brief ExperimentEvaluation
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 15 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#ifndef EXPERIMENTEVALUATION_H
#define EXPERIMENTEVALUATION_H

#include "environment.h"
#include "learning_module/dataformater.h"
#include "feature_extractor/FeatureExtractor.h"
#include "RL_IAC/RegionsM.h"
#include "bottom_up/bottomupsaliencymethod.h"
#include "Evaluation.h"

class ExperimentEvaluation
{
public:
    ExperimentEvaluation();
    ~ExperimentEvaluation();
    bool init_environment(std::string input_path,
                          std::vector<std::string>input_list,
                          std::map<std::string, int> input_region_map,
                          WorldMapGraph::GraphStruct graph_struct);
    bool init_eval_set(int max_samples,
                       int data_resampling_factor,
                       bool data_ignore_unknowns,
                       int subsampling_rate,
                       int evaluation_metrics,
                       bool use_per_region_eval);
    bool init_log_file(string logfilename, string param_filename);
    bool create_eval_set(FeatureExtractor* feature_extractor);
    bool update_log_file(int current_region, RegionsM *region_learner = 0, int time = 0);
    /* For bottom_up evaluation */
    bool create_eval_set(BottomUpSaliencyMethod* bottom_up_sal);

private:
    Environment eval_environment;
    std::vector<DataFormater*> eval_data_formaters;
    std::vector<int> eval_regions;
    int max_samples;
    int data_resampling_factor;
    bool data_ignore_unknowns;
    int subsampling_rate;
    int evaluation_metrics;
    bool use_per_region_eval;
    ofstream * logfile;
    bool HAS_ENVIRONMENT;
    bool HAS_LOGFILE;
    bool HAS_EVALSET;

    /* to switch between classic and bottom_up */
    FeatureExtractor * feature_extractor;
    BottomUpSaliencyMethod * bottom_up_sal;

    bool update_log_file(std::vector<float> score_vect, int current_region, int time = 0);
    bool create_eval_set();
    std::vector<float> get_eval_scores(RegionsM *regions_learner = 0);
};

#endif // EXPERIMENTEVALUATION_H
