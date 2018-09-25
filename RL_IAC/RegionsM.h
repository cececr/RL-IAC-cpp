/**
 * \file RegionsM.h
 * \brief RegionsM
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 15 / 2015
 *
 * TODO: - Add load and save cluster methods
 */

#ifndef RegionsM_H
#define RegionsM_H

#include <cv.h>
#include <vector>
#include "../learning_module/LearningM.h"
#include "../learning_module/dataformater.h"
#include "MetaM.h"

class Region
{
public:
    Region();
    Region(LearningM * learner, int evaluation_metrics = 0, bool use_per_frame_eval = true);
    LearningM *getLearner();
    MetaM *getMetaLearner();
    MetaM *getBackwardMetaLearner();
private:
    LearningM * learner;
    MetaM metaLearner;
    MetaM backwardMetaLearner;
};


class RegionsM
{
public:

    static const int MAX_BACKWARD_SIZE = 10;
    static const int INTR_MOTIV_PROGRESS = 0;
    static const int INTR_MOTIV_NOVELTY = 1;
    static const int INTR_MOTIV_UNCERTAINTY = 2;
    static const int INTR_MOTIV_ERROR = 3;
    static const int INTR_MOTIV_FORGET = 4;
    static const int INTR_MOTIV_PROGRESS_FORGET = 5;

    /* Constructor, destructor and init functions */
    RegionsM();
    virtual ~RegionsM();
    bool init_regions(vector<signed int> regions_learner_list,
                      vector<int> regions_weights,
                      int data_resampling_factor =  24,
                      bool data_ignore_unknowns = true,
                      int max_samples = 0,
                      bool samples_balanced_data =  true,
                      bool samples_random_replace =  true,
                      string classifier_type =  "RandomForest",
                      string classifier_params =  "",
                      int evaluation_metrics = 0,
                      bool use_per_frame_eval = true,
                      bool use_backward = false,
                      int intr_motivation_type = INTR_MOTIV_PROGRESS,
                      float alpha = 0.5);

    /* For long term evaluation */
    void set_long_term_eval_data(vector<cv::Mat> longTermSamples,
                             vector<cv::Mat> longTermClasses,
                             vector<int> longTermRegions);
    void init_long_term_eval(bool use_per_region_eval  = false);
    void add_long_term_samples(int regionIdx, cv::Mat samples, cv::Mat classes);
    void add_long_term_image_data(int regionIdx,
                                 vector<cv::Mat> &feature_maps,
                                 cv::Mat &segmentation_map);

    /* load and save functions */
    bool load_region_model(string model_filename, int regionIdx = 0);
    bool save_region_model(string model_filename, int regionIdx = 0);
    bool save_all_learners(string model_filename);
    bool load_all_learners(string model_filename);

    /* main functions */
    void add_samples(int regionIdx, cv::Mat samples, cv::Mat classes);
    void add_image_samples(int regionIdx, vector<cv::Mat> &feature_maps, cv::Mat &segmentation_map);
    void train(int regionIdx = -1);
    cv::Mat get_saliency_map_at(vector<cv::Mat> feature_map, int regionIdx);
    cv::Mat get_saliency_map_at(vector<cv::Mat> feature_map, cv::Mat superpixels,  int regionIdx);
    cv::Mat get_saliency_estimation_at(cv::Mat samples, int regionIdx);
    float get_progress_at(int regionIdx);
    vector<float> get_regions_progress();
    float get_dataset_fraction(int regionIdx);

    /* for display */
    cv::Mat draw_internal_state(int idxRegion);

private:
    std::vector<Region * > regionsList;
    DataFormater data_formater;
    bool HAS_REGIONS;
    int nb_regions;
    bool use_long_term;
    vector<LearningM*> learner_list;
    vector<signed int> regions_learner_list;
    int intr_motivation_type;
    vector<int> regions_weights;

    // long term evaluation
    bool use_backward;
    bool use_per_region_eval;
    vector<cv::Mat> longTermSamples;
    vector<cv::Mat> longTermClasses;
    vector<int> longTermRegions;
    vector<float> forget_factor;
    float alpha; // only for forget and progress
};

#endif // RegionsM_H
