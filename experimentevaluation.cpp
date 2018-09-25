/**
 * \file experimentevaluation.cpp
 * \brief ExperimentEvaluation
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 15 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#include "experimentevaluation.h"
#include "SaliencyFileParser.h"

ExperimentEvaluation::ExperimentEvaluation()
{
    logfile = 0;
    bottom_up_sal = 0;
    feature_extractor = 0;
    HAS_ENVIRONMENT = false;
    HAS_LOGFILE = false;
    HAS_EVALSET = false;
}

ExperimentEvaluation::~ExperimentEvaluation()
{
    if(logfile != 0)
    {
        logfile->close();
        delete logfile;
    }
    for(int i = 0 ; i < eval_environment.get_nb_regions() ; i++)
    {
        delete eval_data_formaters[i];
    }
}

bool ExperimentEvaluation::init_environment(string input_path,
                                            std::vector<string> input_list,
                                            std::map<string, int> input_region_map,
                                            WorldMapGraph::GraphStruct graph_struct)
{
    eval_environment.init(input_path,input_list, input_region_map, graph_struct);
    HAS_ENVIRONMENT = true;
    return true;
}

bool ExperimentEvaluation::init_eval_set(int max_samples,
                                         int data_resampling_factor,
                                         bool data_ignore_unknowns,
                                         int subsampling_rate,
                                         int evaluation_metrics,
                                         bool use_per_region_eval)
{
    this->max_samples = max_samples;
    this->data_resampling_factor = data_resampling_factor;
    this->data_ignore_unknowns = data_ignore_unknowns;
    this->subsampling_rate = subsampling_rate;
    this->evaluation_metrics = evaluation_metrics;
    this->use_per_region_eval = use_per_region_eval;
    return true;
}


bool ExperimentEvaluation::create_eval_set(FeatureExtractor *feature_extractor)
{
    this->bottom_up_sal = 0;
    this->feature_extractor = feature_extractor;
    return create_eval_set();
}

bool ExperimentEvaluation::create_eval_set(BottomUpSaliencyMethod *bottom_up_sal)
{
    this->bottom_up_sal = bottom_up_sal;
    this->feature_extractor = 0;
    return create_eval_set();
}

bool ExperimentEvaluation::create_eval_set()
{
    assert((feature_extractor == 0 && bottom_up_sal != 0)
           || (feature_extractor != 0 && bottom_up_sal == 0));
    assert(HAS_ENVIRONMENT);
    eval_data_formaters.clear();
    eval_regions.clear();

    int nb_eval_frames = eval_environment.get_inputs_size();
    cv::Mat rgb_input, depth_input, segmentation_mask;
    vector<cv::Mat> feature_map;
    if(use_per_region_eval)
    {
        for(int i = 0 ; i < eval_environment.get_nb_regions() ; i++)
        {
            DataFormater* df = new DataFormater(max_samples,data_resampling_factor,
                                                data_ignore_unknowns);
            eval_data_formaters.push_back(df);
            eval_regions.push_back(i);
        }
    }
    for(int j = 0 ; j < nb_eval_frames ; j = j + subsampling_rate)
    {
        eval_environment.get_inputs(j,rgb_input,depth_input);
        if(feature_extractor != 0)
        {
            feature_map = feature_extractor->getFeatureMap(rgb_input);
        }
        /* the hacks here is to take the bottom up map as the feature map ... */
        else if (bottom_up_sal != 0)
        {
            feature_map.clear();
            feature_map.push_back(bottom_up_sal->getSalMap(rgb_input));
        }
        eval_environment.get_segmentation_input(j,segmentation_mask);
        int region = eval_environment.get_region(j);
        if(use_per_region_eval)
        {
            eval_data_formaters[region]->maps_reformat(feature_map,segmentation_mask);
        }
        else
        {
            DataFormater* df = new DataFormater(max_samples,data_resampling_factor,
                                                data_ignore_unknowns);
            df->maps_reformat(feature_map,segmentation_mask);
            if(df->get_buffers_length() > 0)
            {
                eval_data_formaters.push_back(df);

                eval_regions.push_back(region);
            }
        }

    }
    HAS_EVALSET = true;
    return HAS_EVALSET;

}

std::vector<float> ExperimentEvaluation::get_eval_scores(RegionsM *regions_learner)
{
    vector<float> score_vect;
    if(!HAS_EVALSET)
        return score_vect;

    cv::Mat samples, classes, estimates;
    for(size_t i = 0 ; i < eval_data_formaters.size() ; i++)
    {
        if(regions_learner != 0)
        {
            int regionIdx = eval_regions[i];
            eval_data_formaters[i]->get_buffers(samples, classes);
            estimates = regions_learner->get_saliency_estimation_at(samples,regionIdx);
        }
        else
        {
            /* trick: take samples as estimates */
            eval_data_formaters[i]->get_buffers(estimates, classes);
        }

        if(classes.empty())
        {
            score_vect.push_back(NAN);
            continue;
        }

        score_vect.push_back(Eval::evaluate(classes,estimates,evaluation_metrics));
    }
    return score_vect;
}

bool ExperimentEvaluation::init_log_file(string logfilename, string param_filename)
{
    logfile = new ofstream(logfilename.c_str(), ios::out | ios::trunc);
    if(!logfile)
        return false;

    SaliencyFileParser fp;
    fp.saveInFile(*logfile, param_filename);
    SaliencyFileParser::Param_Struct log_params;
    bool res = fp.parse_param_file(param_filename,log_params);
    typedef std::map<std::string, std::string>::iterator it_type;
    for(it_type iterator = log_params.string.begin(); iterator != log_params.string.end(); iterator++)
    {
        if(iterator->second.find(".yaml") != std::string::npos)
        {
            res = res & fp.saveInFile(*logfile, iterator->second);
        }
    }
    HAS_LOGFILE = res;
    return res;
}

bool ExperimentEvaluation::update_log_file(int current_region, RegionsM *region_learner, int time)
{
    if(!HAS_LOGFILE)
        return false;
    /* Evaluate current model from selected frames */
    vector<float> score_vect = get_eval_scores(region_learner);
    return update_log_file(score_vect,current_region, time);
}


bool ExperimentEvaluation::update_log_file(std::vector<float> score_vect, int current_region, int time)
{
    cv::Mat score_per_region = cv::Mat::zeros(1,eval_environment.get_nb_regions(),CV_32F);
    cv::Mat nb_per_region = cv::Mat::zeros(1,eval_environment.get_nb_regions(),CV_32F);
    for(size_t i = 0 ; i < score_vect.size() ; i++)
    {
        if(isnan(score_vect[i]))
            continue;
        score_per_region.at<float>(eval_regions[i]) += score_vect[i];
        nb_per_region.at<float>(eval_regions[i]) += 1;
    }
    *logfile << "Region scores" << endl ;
    for(int i = 0 ; i < score_per_region.cols ; i++)
    {
        *logfile << score_per_region.at<float>(i)/nb_per_region.at<float>(i) << " ";
    }
    *logfile << endl ;
    *logfile << "Current region = " << current_region << endl;
    *logfile << "Time = " << time << endl;
    return true;
}




