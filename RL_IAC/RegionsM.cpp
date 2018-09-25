#include "RegionsM.h"
#include "printdebug.h"
#include <algorithm>
#include "common.h"

using namespace Common;
using namespace cv;


RegionsM::RegionsM()
{
    HAS_REGIONS = false;
    nb_regions = 0;
    use_long_term = false;
    use_backward = false;
}

RegionsM::~RegionsM()
{
    for(int i = 0 ; i < nb_regions ; i++)
    {
        delete regionsList[i];
    }
}

bool RegionsM::init_regions(vector<signed int> regions_learner_list,
                            vector<int> regions_weights,
                            int data_resampling_factor,
                            bool data_ignore_unknowns,
                            int max_samples,
                            bool samples_balanced_data,
                            bool samples_random_replace,
                            string classifier_type,
                            string classifier_params,
                            int evaluation_metrics,
                            bool use_per_frame_eval,
                            bool use_backward,
                            int intr_motivation_type,
                            float alpha)
{
    /* Initialize variables */
    HAS_REGIONS = false;
    use_long_term = false;
    this->intr_motivation_type = intr_motivation_type;
    this->use_backward = use_backward;
    if(use_backward)
        init_long_term_eval(!use_per_frame_eval);
    nb_regions = regions_learner_list.size();
    if (nb_regions == 0)
        return false;
    this->regions_learner_list = regions_learner_list;
    int learningM_type = 0;
    if(classifier_type == "RandomForest")
        learningM_type = LearningM::My_RANDOM_TREE;
    if(classifier_type == "NeuralNet")
        learningM_type = LearningM::NEURAL_NET;
    if(classifier_type == "OnlineRandomForest")
        learningM_type = LearningM::ONLINE_RANDOM_TREE;
    else
        learningM_type = LearningM::My_RANDOM_TREE;

    /* Initialize data formater */
    data_formater = DataFormater(max_samples, data_resampling_factor, data_ignore_unknowns);

    /* Initialize learners */
    int nb_learner = *max_element(regions_learner_list.begin(), regions_learner_list.end());
//    vector<LearningM*> learner_list;
    for(int i = 0 ; i < nb_learner+1 ; i++)
    {
        LearningM * learner = new LearningM();
        learner->init(learningM_type, max_samples, samples_balanced_data,
                      samples_random_replace,classifier_params);
        learner_list.push_back(learner);
    }

    /* Initialize regions */
    for(int i = 0 ; i < nb_regions ; i++)
    {
        Region* region = new Region(learner_list[regions_learner_list[i]],
                         evaluation_metrics, use_per_frame_eval);
        regionsList.push_back(region);
    }

    /* Initialize forgetting factor */
    for(int i = 0 ; i < nb_regions ; i++)
    {
        forget_factor.push_back(1);
    }

    this->alpha = alpha;
    assert(regions_weights.size() == regions_learner_list.size());
    this->regions_weights = regions_weights;

    HAS_REGIONS = true;
    return true;
}

void RegionsM::add_samples(int regionIdx, Mat samples, Mat classes)
{
    /* Sanity checks */
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    if(!HAS_REGIONS)
        return;

//    if(samples.empty() ||classes.empty())
//    {
//        // TODO: Do not return here ... Add max possible error
//        return;
//    }

    /* Adding samples to the learner */
    if(!(samples.empty() ||classes.empty()))
    {
        samples.convertTo(samples, CV_32F);
        classes.convertTo(classes, CV_32F);
        regionsList[regionIdx]->getLearner()->addBatchSamples(samples,classes);
    }

    /* Updating forgetting factor */
    for(int i = 0 ; i < nb_regions ; i++)
    {
        if(i == regionIdx)
        {
            forget_factor[i] = 1;
        }
        else
        {
            forget_factor[i]*=0.96;// TODO: set lambda
        }
    }

    /* Adding samples to the meta learner */
    if(!use_long_term) // Regular error evaluation
    {
        Mat estimates;
        if(!(samples.empty() || classes.empty()))
        {
            /* Predict saliency on samples */
            estimates = regionsList[regionIdx]->getLearner()->predict(samples);
        }
        else
        {
            /* set worse case */
            estimates = Mat::zeros(1,MetaM::INPUT_DATA_SIZE,CV_32F);
            classes = Mat::ones(1,MetaM::INPUT_DATA_SIZE,CV_32F);
        }

        /* Update meta learner */
        regionsList[regionIdx]->getMetaLearner()->updateErrors(estimates, classes);
    }
    if(use_long_term || use_backward)
    {
        if(use_backward && !(samples.empty()))
        {
            /* Update backward meta learner sets */
            add_long_term_samples(regionIdx, samples, classes);
        }

        int updated_learner = regions_learner_list[regionIdx];
        for(size_t i = 0 ; i < longTermRegions.size() ; i++)
        {
            /* If learner is shared with current region, update is required */
            int long_term_region = longTermRegions[i];
            if(regions_learner_list[long_term_region] != updated_learner)
                continue;
            /* Predict saliency on samples */
            Mat estimates = regionsList[long_term_region]
                            ->getLearner()->predict(longTermSamples[i]);
            /* Update meta learner */
            MetaM* meta_learner;
            if(use_backward)
                meta_learner = regionsList[long_term_region]->getBackwardMetaLearner();
            else
                meta_learner = regionsList[long_term_region]->getMetaLearner();

            if(!longTermClasses[i].empty())
            {

               meta_learner->updateErrors(estimates, longTermClasses[i]);
            }
            else
            {
                meta_learner->updateErrors(cv::Mat::zeros(1,1,CV_32F),
                                       cv::Mat::ones(1,1,CV_32F));
            }
        }
    }
}

void RegionsM::add_image_samples(int regionIdx, vector<Mat> & feature_maps, Mat &segmentation_map)
{
    data_formater.maps_reformat(feature_maps, segmentation_map);
    Mat samples, classes;
    data_formater.get_buffers(samples,classes);
    data_formater.clear_buffers();
    add_samples(regionIdx,samples,classes);
}

void RegionsM::train(int regionIdx)
{
    if(regionIdx < 0 || regionIdx > nb_regions )
    {
        /* train all */
        for(size_t i = 0 ; i < learner_list.size() ; i++)
        {
            std::cout << "training learner " << i+1 << " of " << learner_list.size() << std::endl;
            learner_list[i]->train();
        }
    }
    else
    {
        /* train region only */
        regionsList[regionIdx]->getLearner()->train();
    }
}

Mat RegionsM::get_saliency_map_at(vector<Mat> feature_map, int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    LearningM * learner = regionsList[regionIdx]->getLearner();
    return learner->estimateSaliencyFromFeatureMap(feature_map);
}

Mat RegionsM::get_saliency_map_at(vector<Mat> feature_map, Mat superpixels, int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    LearningM * learner = regionsList[regionIdx]->getLearner();
    return learner->estimateSaliencyFromFeatureMap(feature_map, superpixels);
}

Mat RegionsM::get_saliency_estimation_at(Mat samples, int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    LearningM * learner = regionsList[regionIdx]->getLearner();
    return learner->predict(samples);
}

float RegionsM::get_progress_at(int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    MetaM* meta_learner;
    if(use_backward)
        meta_learner = regionsList[regionIdx]->getBackwardMetaLearner();
    else
        meta_learner = regionsList[regionIdx]->getMetaLearner();

    float progress;
    //float alpha = 0.1; // TODO: set paramemet alpha to control influence
    switch(intr_motivation_type)
    {
        case INTR_MOTIV_PROGRESS:
            progress = abs(meta_learner->getLearningProgress())*regions_weights[regionIdx];
            break;
        case INTR_MOTIV_ERROR:
            progress = meta_learner->getLearningError();
            break;
        case INTR_MOTIV_UNCERTAINTY:
            progress = meta_learner->getLearningUncertainty();
            break;
        case INTR_MOTIV_NOVELTY:
            progress = get_dataset_fraction(regionIdx);
            break;
        case INTR_MOTIV_FORGET:
            progress = 1-forget_factor[regionIdx];
            break;
        case INTR_MOTIV_PROGRESS_FORGET:
            cout << "region " << regionIdx <<" LP=" << 2*atan(meta_learner->getLearningProgress())/CV_PI
                 << " forget_factor " << forget_factor[regionIdx]
                 << " regions_weights " << regions_weights[regionIdx] << endl;
            if(meta_learner->getNsamples()<3 && meta_learner->getLearningProgress() != 0)
                progress = 100;
            else
            {
                progress = (1-alpha)*2*atan(meta_learner->getLearningProgress())/CV_PI
                     + alpha*(1-forget_factor[regionIdx]);//*(float)regions_weights[regionIdx]/(float)sum(regions_weights).val[0];
            }
            cout << "progress " << progress << endl;
            break;
        default:
            progress = 0;
    }

    return progress;
}

vector<float> RegionsM::get_regions_progress()
{
    vector<float> progress_list;
    for(int i = 0 ; i < nb_regions ; i++)
    {
        progress_list.push_back(get_progress_at(i));
    }
    return progress_list;
}

float RegionsM::get_dataset_fraction(int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    MetaM* meta_learner;
    int max_nb_samples = 0;
    int min_nb_samples = -1;
    int samples_in_region = 0;
    for(int i = 0 ; i < nb_regions ; i++)
    {
        if(use_backward)
            meta_learner = regionsList[i]->getBackwardMetaLearner();
        else
            meta_learner = regionsList[i]->getMetaLearner();

        int nb_samples = meta_learner->getNsamples();
        if(nb_samples > max_nb_samples)
            max_nb_samples = nb_samples;
        if(min_nb_samples < 0 || nb_samples < min_nb_samples)
            min_nb_samples = nb_samples;
        if(i == regionIdx)
            samples_in_region = nb_samples;
    }
    return (float)min_nb_samples == samples_in_region;
}

Mat RegionsM::draw_internal_state(int idxRegion)
{
    vector<Plot> plotList;
    Mat subPlotMat;
    MetaM* meta_learner;


    for(int i = 0 ; i < nb_regions ; i++)
    {
        if(use_backward)
            meta_learner = regionsList[i]->getBackwardMetaLearner();
        else
            meta_learner = regionsList[i]->getMetaLearner();

        Plot regionErrPlot = meta_learner->displayClusterError();
        if(regionErrPlot.getDataSize().width > 0)
        {
            std::ostringstream ss;
            ss << 2*atan(meta_learner->getLearningProgress())/CV_PI;
            std::string s(ss.str());
            string str = "Region # " + to_string(i)
                    + " nsamples = " + to_string(meta_learner->getNsamples())
                    + " Progress = " + s;
            if(i == idxRegion)
            {
                regionErrPlot.frameOn();
            }
            regionErrPlot.setTitle(str);
            regionErrPlot.setLineStyle(Plot::line);
            regionErrPlot.setyrange(0,1);
            plotList.push_back( regionErrPlot );
        }
    }
    if(plotList.size()> 0)
    {
        Subplot sub(plotList);
        subPlotMat = sub.draw();
    }
    return subPlotMat;
}

void RegionsM::set_long_term_eval_data(vector<Mat> longTermSamples,
                                       vector<Mat> longTermClasses,
                                       vector<int> longTermRegions)
{
    assert(!use_backward);
    assert(longTermSamples.size() > 0
           && longTermSamples.size() == longTermClasses.size()
           && longTermSamples.size() == longTermRegions.size());


    /* check that provided regions are consistent */
    int  max_el = *std::max_element(longTermRegions.begin(), longTermRegions.end());
    int  min_el = *std::min_element(longTermRegions.begin(), longTermRegions.end());
    assert(max_el<nb_regions
           && min_el >= 0);

    this->longTermSamples = longTermSamples;
    this->longTermClasses = longTermClasses;
    this->longTermRegions = longTermRegions;
    use_long_term = true;
}

void RegionsM::init_long_term_eval(bool use_per_region_eval)
{
    this->use_per_region_eval = use_per_region_eval;
    if(use_per_region_eval)
    {
        longTermSamples.resize(nb_regions);
        longTermClasses.resize(nb_regions);
        for(int i = 0 ; i < nb_regions ; i++)
            longTermRegions.push_back(i);
    }
    if(!use_backward)
        use_long_term = true;
}

void RegionsM::add_long_term_samples(int regionIdx, Mat samples, Mat classes)
{
    assert(use_backward);
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    if(samples.empty()|| classes.empty())
        return;

    if(use_per_region_eval)
    {
        if(longTermSamples[regionIdx].rows > samples.rows*MAX_BACKWARD_SIZE)
        {
            srand(time(NULL));
            int idx = rand()% longTermSamples[regionIdx].rows-samples.rows;
            Rect r(0,idx,samples.cols, samples.rows);
            samples.copyTo(longTermSamples[regionIdx](r));
            classes.copyTo(longTermClasses[regionIdx](r));
        }
        else
        {
            longTermSamples[regionIdx].push_back(samples);
            longTermClasses[regionIdx].push_back(classes);
        }
    }
    else
    {
        /* find indices for which regionsList = regionIdx */
        vector<int> regionIdxIndices;
        for(size_t i = 0 ; i < longTermRegions.size() ; i++)
        {
            if(longTermRegions[i] == regionIdx)
                regionIdxIndices.push_back(i);
        }

        if(regionIdxIndices.size()>MAX_BACKWARD_SIZE)
        {
            srand(time(NULL));
            int idx = rand()% regionIdxIndices.size();
            longTermSamples[regionIdxIndices[idx]] = samples;
            longTermClasses[regionIdxIndices[idx]] = classes;
            longTermRegions[regionIdxIndices[idx]] = regionIdx;
        }
        else
        {
            longTermSamples.push_back(samples);
            longTermClasses.push_back(classes);
            longTermRegions.push_back(regionIdx);
        }
        regionsList[regionIdx]->getBackwardMetaLearner()->setWindow(10*MAX_BACKWARD_SIZE);
        regionsList[regionIdx]->getBackwardMetaLearner()->setSmooth(20*MAX_BACKWARD_SIZE);
    }
}

void RegionsM::add_long_term_image_data(int regionIdx,
                                       vector<Mat> &feature_maps,
                                       Mat &segmentation_map)
{
    data_formater.maps_reformat(feature_maps, segmentation_map);
    Mat samples, classes;
    data_formater.get_buffers(samples,classes);
    data_formater.clear_buffers();
    add_long_term_samples(regionIdx,samples,classes);
}

bool RegionsM::load_region_model(string model_filename, int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    return regionsList[regionIdx]->getLearner()->load(model_filename.c_str());
}

bool RegionsM::save_region_model(string model_filename, int regionIdx)
{
    assert(regionIdx >= 0 && regionIdx < nb_regions);
    return regionsList[regionIdx]->getLearner()->save(model_filename.c_str());
}

bool RegionsM::save_all_learners(string model_filename)
{
    bool res = true;
    for(size_t i = 0 ; i < learner_list.size() ; i++)
    {
        std::cout << "saving model " << i+1 << " of " << learner_list.size() << std::endl;
        string learner_filename = model_filename;
        if(learner_filename.find(".yml") == std::string::npos)
        {
            learner_filename = learner_filename + ".yml";
        }
        if(learner_list.size()> 1)
        {
            string learner_idx = "_learner" + to_string(i);
            learner_filename.insert(learner_filename.length()-4,learner_idx);
        }
        std::cout << "saving model " << learner_filename << std::endl;
        res = res & learner_list[i]->save(learner_filename.c_str());
    }
    return res;
}

bool RegionsM::load_all_learners(string model_filename)
{
    if(!HAS_REGIONS)
        return false;

    bool res = true;
    /* Try to match learners with available models */
    for(size_t i = 0 ; i < learner_list.size() ; i++)
    {
        string learner_filename = model_filename;
        if(learner_filename.find(".yml") == std::string::npos)
        {
            learner_filename = learner_filename + ".yml";
        }

        if(learner_list.size() ==  1)
        {
            if(!learner_list[i]->load(learner_filename.c_str()))
            {
                string learner_idx = "_learner" + to_string(i);
                learner_filename.insert(learner_filename.length()-4,learner_idx);
            }
        }
        else
        {
            string learner_idx = "_learner" + to_string(i);
            learner_filename.insert(learner_filename.length()-4,learner_idx);
        }
        res = res & learner_list[i]->load(learner_filename.c_str());
    }
    return res;
}

//////////////////////////////////////////////////////
/// Region Class
///


Region::Region(LearningM* learner, int evaluation_metrics, bool use_per_frame_eval)
{
    this->learner = learner;
    metaLearner = MetaM(evaluation_metrics, use_per_frame_eval);
    backwardMetaLearner = MetaM(evaluation_metrics, use_per_frame_eval);
}

LearningM* Region::getLearner()
{
    return learner;
}

MetaM* Region::getMetaLearner()
{
    return &metaLearner;
}

MetaM *Region::getBackwardMetaLearner()
{
    return &backwardMetaLearner;
}

