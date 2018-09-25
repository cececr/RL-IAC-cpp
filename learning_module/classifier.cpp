/**
 * \file classifier.cpp
 * \brief classifier
 * \author Céline Craye
 * \version 0.1
 * \date 2014
 *
 * Please check header files for a detailed description
 *
 */
#include "classifier.h"
using namespace cv;

/**
 * @brief Classifier::Classifier
 */
Classifier::Classifier()
{
    TRAINED = false;
    INIT = false;
    SINGLEVALUE = -1;
}

/**
 * @brief Classifier::~Classifier
 */
Classifier::~Classifier()
{
}

/**
 * @brief Classifier::isTrained
 * @return
 */
bool Classifier::isTrained()
{
    return TRAINED;
}

/**
 * @brief Classifier::isInit
 * @return
 */
bool Classifier::isInit()
{
    return INIT;
}

/**
 * @brief Classifier::getSingleValue
 * @return
 */
int Classifier::getSingleValue()
{
    return SINGLEVALUE;
}

/**
 * @brief RFClassifier::init
 * @param param_file
 * @return
 */
bool RFClassifier::init(string param_file)
{
    // Try to load parameters
    if(!load_params(param_file.c_str()))
    {
        if(!param_file.empty())
        {
            std::cout << "Could not load param file " << param_file << std::endl;
            std::cout << "Loading default parameters" << std::endl;
        }

        /* Use default params */
        float* priors = new float[10];
        for(int i = 0 ; i < 10 ; i++)
            priors[i] = 1;

        rf_params = CvRTParams(10,//10, // max depth
                               5, //5 min sample count
                               0, // regression accuracy: N/A here
                               false, // compute surrogate split, no missing data
                               15, // max number of categories (use sub-optimal algorithm for larger numbers)
                               priors, // the array of priors
                               false,  // calculate variable importance
                               4,       // number of variables randomly selected at node and used to find the best split(s).
                               20,//100,	 // max number of trees in the forest
                               0.01f,				// forest accuracy
                               CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                              );
    }

    INIT = true;
    return true;
}

/**
 * @brief RFClassifier::load
 * @param filename
 * @return
 */
bool RFClassifier::load(const char *filename)
{
    /* Load param files */
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!load_params(strfilename.c_str()))
        return false;

    /* Load model */
    TRAINED = false;
    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    FileStorage fs(strfilename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs.release();
    rf.load(strfilename.c_str());
    TRAINED = true;
    return true;
}

/**
 * @brief RFClassifier::save
 * @param filename
 * @return
 */
bool RFClassifier::save(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!INIT)
        return false;
    save_params(strfilename.c_str());

    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    if(!TRAINED)
        return false;
    rf.save(strfilename.c_str());

    return true;
}

/**
 * @brief RFClassifier::train
 * @param samples
 * @param classes
 * @return
 */
bool RFClassifier::train(Mat &samples, Mat &classes)
{
    cout << "Training Random Forest ..." << endl;

    double min, max;
    minMaxLoc(classes,&min,&max);
    if(min == max) // only one class
    {
        TRAINED = false;
    }
    else
    {
        //PrintDebug::print(samples, "RF samples");
        //PrintDebug::print(classes, "RF classes");
        rf.train(samples, CV_ROW_SAMPLE,classes, cv::Mat(), cv::Mat(), Mat(), Mat(),rf_params);
        cout << "Finished !" << endl;
        TRAINED = true;
    }
    return TRAINED;
}

/**
 * @brief RFClassifier::predict
 * @param samples
 * @param useprobs
 * @return
 */
Mat RFClassifier::predict(Mat samples,bool useprobs)
{
    /* Make sure input data is floating points matrix */
    samples.convertTo(samples, CV_32F);

    /* Create output matrix */
    int nrows = samples.rows;
    int  ncols = 1;
    Mat predictedLabels = Mat::zeros(nrows, ncols, samples.type());
    if(TRAINED)
    {
        for(int i= 0 ; i < nrows ; i++)
        {
            if(useprobs)
                predictedLabels.at<float>(i) = rf.predict_prob(samples.row(i));
            else
                predictedLabels.at<float>(i) = rf.predict(samples.row(i));
        }
    }

    return predictedLabels;
}

/**
 * @brief RFClassifier::save_params
 * @param filename
 * @return
 */
bool RFClassifier::save_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "priors" << Mat(vector<float>(rf_params.priors,rf_params.priors+10));
    fs << "max_depth" << rf_params.max_depth;
    fs << "min_sample_count" << rf_params.min_sample_count;
    fs << "regression_accuracy" << rf_params.regression_accuracy;
    fs << "use_surrogates" << rf_params.use_surrogates;
    fs << "max_categories" << rf_params.max_categories;
    fs << "calc_var_importance" << rf_params.calc_var_importance;
    fs << "nactive_vars" << rf_params.nactive_vars;
    fs << "max_num_of_trees_in_the_forest" << rf_params.term_crit.max_iter;
    fs << "forest_accuracy" << rf_params.term_crit.epsilon;
    fs << "termcrit_type" << rf_params.term_crit.type;


    fs.release();
    return true;
}

/**
 * @brief RFClassifier::load_params
 * @param filename
 * @return
 */
bool RFClassifier::load_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::READ);
    INIT = false;
    if(!fs.isOpened())
        return false;

    Mat priorsMat;
    fs["priors"] >> priorsMat;
    float* priors = (float*) priorsMat.data;
    rf_params = CvRTParams((int) fs["max_depth"],
                           (int) fs["min_sample_count"],
                           (int) fs["regression_accuracy"],
                           (int) fs["use_surrogates"],
                           (int) fs["max_categories"],
                           priors, // the array of priors
                           (int) fs["calc_var_importance"],
                           (int) fs["nactive_vars"],
                           (int) fs["max_num_of_trees_in_the_forest"],
                           (float) fs["forest_accuracy"],
                           (int) fs["termcrit_type"]);

    INIT = true;
    fs.release();
    return true;
}

/////////////////////////////////////////////////////////////////////
/// My Random Forest Classifier
///
/**
 * @brief MyRFClassifier::init
 * @param param_file
 * @return
 */
bool MyRFClassifier::init(std::string param_file)
{
    // Try to load parameters
    if(!load_params(param_file.c_str()))
    {
        if(!param_file.empty())
        {
            std::cout << "Could not load param file " << param_file << std::endl;
            std::cout << "Loading default parameters" << std::endl;
        }
        /* Provide params */
        float* priors = new float[10];
        for(int i = 0 ; i < 10 ; i++)
            priors[i] = 1;

        rf_params = RTParams(10,//10, // max depth
                               10, //5 min sample count
                               0, // regression accuracy: N/A here
                               false, // compute surrogate split, no missing data
                               15, // max number of categories (use sub-optimal algorithm for larger numbers)
                               priors, // the array of priors
                               false,  // calculate variable importance
                               4,       // number of variables randomly selected at node and used to find the best split(s).
                               30,//20,	 // max number of trees in the forest
                               0.01f,				// forest accuracy
                               CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                              );
    }
    INIT = true;
    return true;
}

/**
 * @brief MyRFClassifier::load
 * @param filename
 * @return
 */
bool MyRFClassifier::load(const char *filename)
{
    /* Load param files */
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!load_params(strfilename.c_str()))
        return false;

    /* Load model */
    TRAINED = false;
    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    FileStorage fs(strfilename.c_str(), FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs.release();
    rf.load(strfilename.c_str());
    TRAINED = true;
    return true;
}

/**
 * @brief MyRFClassifier::save
 * @param filename
 * @return
 */
bool MyRFClassifier::save(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!INIT)
        return false;
    save_params(strfilename.c_str());

    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    if(!TRAINED)
        return false;
    rf.save(strfilename.c_str());

    return true;
}

/**
 * @brief MyRFClassifier::save_params
 * @param filename
 * @return
 */
bool MyRFClassifier::save_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if(!fs.isOpened())
        return false;
    if(!INIT)
        return false;

    fs << "priors" << Mat(vector<float>(rf_params.priors,rf_params.priors+10));
    fs << "max_depth" << rf_params.max_depth;
    fs << "min_sample_count" << rf_params.min_sample_count;
    fs << "regression_accuracy" << rf_params.regression_accuracy;
    fs << "use_surrogates" << rf_params.use_surrogates;
    fs << "max_categories" << rf_params.max_categories;
    fs << "calc_var_importance" << rf_params.calc_var_importance;
    fs << "nactive_vars" << rf_params.nactive_vars;
    fs << "max_num_of_trees_in_the_forest" << rf_params.term_crit.max_iter;
    fs << "forest_accuracy" << rf_params.term_crit.epsilon;
    fs << "termcrit_type" << rf_params.term_crit.type;


    fs.release();
    return true;
}

/**
 * @brief MyRFClassifier::load_params
 * @param filename
 * @return
 */
bool MyRFClassifier::load_params(const char *filename)
{
    INIT = false;
    if((filename != NULL) && (filename[0] == '\0'))
        return false;
    FileStorage fs(filename, FileStorage::READ);

    if(!fs.isOpened())
        return false;

    Mat priorsMat;
    fs["priors"] >> priorsMat;

    float* priors = new float[priorsMat.rows];
    for(int i = 0 ;  i < priorsMat.rows ; i++)
    {
        priors[i] = priorsMat.at<float>(i);
    }

    rf_params = RTParams((int) fs["max_depth"],
                         (int) fs["min_sample_count"],
                         (int) fs["regression_accuracy"],
                         (int) fs["use_surrogates"],
                         (int) fs["max_categories"],
                         priors,
                         (int) fs["calc_var_importance"],
                         (int) fs["nactive_vars"],
                         (int) fs["max_num_of_trees_in_the_forest"],
                         (float) fs["forest_accuracy"],
                         (int) fs["termcrit_type"]);

    INIT = true;
    fs.release();
    return true;
}

/**
 * @brief MyRFClassifier::train
 * @param samples
 * @param classes
 * @return
 */
bool MyRFClassifier::train(Mat &samples, Mat &classes)
{
    cout << "Training Random Forest ..." << endl;
    if(samples.empty() || classes.empty())
    {
        TRAINED = false;
        return TRAINED;
    }

    double min, max;
    minMaxLoc(classes,&min,&max);
    if(min == max) // only one class
    {
        TRAINED = false;
        SINGLEVALUE = min;
    }
    else
    {
        rf.train(samples, CV_ROW_SAMPLE,classes, cv::Mat(), cv::Mat(), Mat(), Mat(),rf_params);
        cout << "Done." << endl;

        TRAINED = true;
    }
    return TRAINED;
}

/**
 * @brief MyRFClassifier::predict
 * @param samples
 * @param useprobs
 * @return
 */
Mat MyRFClassifier::predict(Mat samples,bool useprobs)
{
    /* Make sure input data is floating points matrix */
    samples.convertTo(samples, CV_32F);

    /* Create output matrix */
    int nrows = samples.rows;
    int  ncols = 1;
    Mat predictedLabels = Mat::zeros(nrows, ncols, samples.type());
    if(TRAINED)
    {
        for(int i= 0 ; i < nrows ; i++)
        {
            if(useprobs)
                predictedLabels.at<float>(i) = rf.predict_prob(samples.row(i));
            else
                predictedLabels.at<float>(i) = rf.predict(samples.row(i));
        }
    }
    else
    {
        predictedLabels = predictedLabels * SINGLEVALUE;
    }
    return predictedLabels;
}

/**
 * @brief MyRFClassifier::noveltyScore
 * @param samples
 * @return
 */
Mat MyRFClassifier::noveltyScore(Mat samples)
{
    /* Make sure input data is floating points matrix */
    samples.convertTo(samples, CV_32F);

    /* Create output matrix */
    int nrows = samples.rows;
    int  ncols = 1;
    Mat noveltyScores(nrows, ncols, samples.type());
    for(int i= 0 ; i < nrows ; i++)
    {
        CvMat sample = samples.row(i);
        noveltyScores.at<float>(i) = rf.getNovelty(&sample);
//		noveltyScores.at<float>(i) = rf.getBinUncertainty(&sample);
    }

    return noveltyScores;
}

/**
 * @brief MyRFClassifier::initClusters
 * @param trainData
 * @param K
 */
void MyRFClassifier::initClusters(Mat trainData, int K)
{
    rf.makeClusters(trainData, K);
}

Mat MyRFClassifier::getClusters(Mat samples)
{
    Mat clusterList = Mat::zeros(samples.rows,1,CV_32S);
    CvMat sample;
    for(int i = 0 ; i < samples.rows ; i++)
    {
        sample = samples.row(i);
        clusterList.at<int>(i) = rf.getCluster(&sample);
    }
    //PrintDebug::print(clusterList,"cluster list");
    return clusterList;
}


/////////////////////////////////////////////////////////////////////
/// MLP Classifier
///

/**
 * @brief MLPClassifier::init
 * @param param_file
 * @return
 */
bool MLPClassifier::init(string param_file)
{
    // Try to load parameters
    if(!load_params(param_file.c_str()))
    {
        if(!param_file.empty())
        {
            std::cout << "Could not load param file " << param_file << std::endl;
            std::cout << "Loading default parameters" << std::endl;
        }
        /* Provide params */
        CvTermCriteria criteria;
        criteria.max_iter = 5000;
        criteria.epsilon = 0.00001f;
        criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
        mlp_params.train_method = CvANN_MLP_TrainParams::BACKPROP;//RPROP
        mlp_params.bp_dw_scale = 0.05f;
        mlp_params.bp_moment_scale = 0.05f;
        mlp_params.term_crit = criteria;
    }

    INIT = true;
    return true;
}

/**
 * @brief MLPClassifier::load
 * @param filename
 * @return
 */
bool MLPClassifier::load(const char *filename)
{
    /* Load param files */
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!load_params(strfilename.c_str()))
        return false;

    /* Load model */
    TRAINED = false;
    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    FileStorage fs(strfilename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs.release();
    mlp.load(strfilename.c_str());
    TRAINED = true;
    return true;
}

/**
 * @brief MLPClassifier::save
 * @param filename
 * @return
 */
bool MLPClassifier::save(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!INIT)
        return false;
    save_params(strfilename.c_str());

    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    if(!TRAINED)
        return false;
    mlp.save(strfilename.c_str());

    return true;
}

/**
 * @brief MLPClassifier::train
 * @param samples
 * @param classes
 * @return
 */
bool MLPClassifier::train(Mat &samples, Mat &classes)
{
    /* Design layers */
    cv::Mat layers = cv::Mat(3, 1, CV_32SC1);
    layers.row(0) = cv::Scalar(samples.cols);
    layers.row(1) = cv::Scalar(30);
    layers.row(2) = cv::Scalar(2); // Two classes problems only
    /* Create MLP */
    mlp.create(layers);

    /* Convert classes to appropriate matrix */
    double min, max;
    minMaxLoc(classes,&min,&max);
    if(min == max || max > 2) // only one class
    {
        TRAINED = false;
        return false;
    }
    Mat formatedClasses = Mat::zeros(samples.rows,2,CV_32F);
    for(int i = 0 ; i < samples.rows ; i++)
        formatedClasses.at<float>(i,classes.at<float>(0,i)) = 1;

    cout << "Training MLP ..." << endl;

    /* Convert classes to fit the neural net */

    mlp.train(samples, formatedClasses, cv::Mat(), cv::Mat(), mlp_params);
    cout << "Finished !" << endl;
    TRAINED = true;
    return TRAINED;
}

/**
 * @brief MLPClassifier::predict
 * @param samples
 * @param useprobs
 * @return
 */
Mat MLPClassifier::predict(Mat samples,bool useprobs)
{
    /* Make sure input data is floating points matrix */
    samples.convertTo(samples, CV_32F);

    /* Create output matrix */
    int nrows = samples.rows;
    int  ncols = *((int *)mlp.get_layer_sizes()->data.ptr + mlp.get_layer_sizes()->cols -1);
    Mat predictedLabels = Mat::zeros(nrows, ncols, samples.type());
    if(TRAINED)
    {
        Mat formatedPredictedLabels = Mat::zeros(nrows,1,CV_32F);

        /* predict */
        mlp.predict(samples,predictedLabels);

        /* Reformat */
        for(int i = 0 ; i < nrows ; i++)
        {
            double min, max;
            Point minL,maxL;
            minMaxLoc(predictedLabels(Rect(0,i,ncols,1)),&min,&max,&minL,&maxL);
            if(useprobs)
            {
                formatedPredictedLabels.at<float>(i,0) = maxL.x; //TODO: make it smooth
            }
            else
            {
                formatedPredictedLabels.at<float>(i,0) = maxL.x;
            }
        }
    }
    return predictedLabels;
}

/**
 * @brief MLPClassifier::save_params
 * @param filename
 * @return
 */
bool MLPClassifier::save_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::WRITE);

    fs << "train_method" << mlp_params.train_method;
    fs << "bp_dw_scale" << mlp_params.bp_dw_scale;
    fs << "bp_moment_scale" << mlp_params.bp_moment_scale;
    fs << "max_iter" << mlp_params.term_crit.max_iter;
    fs << "epsilon" << mlp_params.term_crit.epsilon;
    fs << "type" << mlp_params.term_crit.type;

    fs.release();
    return true;
}

/**
 * @brief MLPClassifier::load_params
 * @param filename
 * @return
 */
bool MLPClassifier::load_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::READ);

    INIT = false;
    if(!fs.isOpened())
        return false;

    mlp_params.train_method = (int) fs["train_method"];
    mlp_params.bp_dw_scale = (float) fs["bp_dw_scale"];
    mlp_params.bp_moment_scale = (float) fs["bp_moment_scale"];
    CvTermCriteria criteria;
    criteria.max_iter = (int) fs["max_iter"];
    criteria.epsilon = (float) fs["epsilon"];
    criteria.type = (int) fs["type"];
    mlp_params.term_crit = criteria;

    INIT = true;
    fs.release();
    return true;
}

/////////////////////////////////////////////////////////////
/// ONLINE RANDOM FOREST CLASSIFIER
///
/**
 * @brief OnlineRFClassifier::init
 * @param param_file
 * @return
 */
bool OnlineRFClassifier::init(string param_file)
{
    if(!load_params(param_file.c_str()))
    {
        if(!param_file.empty())
        {
            std::cout << "Could not load param file " << param_file << std::endl;
            std::cout << "Loading default parameters" << std::endl;
        }
        /* Provide params */
        float* priors = new float[10];
        for(int i = 0 ; i < 10 ; i++)
            priors[i] = 1;

        CvRTParams rf_params = CvRTParams(10,//10, // max depth
                               10, //5 min sample count
                               0, // regression accuracy: N/A here
                               false, // compute surrogate split, no missing data
                               15, // max number of categories (use sub-optimal algorithm for larger numbers)
                               priors, // the array of priors
                               false,  // calculate variable importance
                               4,       // number of variables randomly selected at node and used to find the best split(s).
                               30,//20,	 // max number of trees in the forest
                               0.01f,				// forest accuracy
                               CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                              );
        INIT = rf.init(rf_params);
    }
    return INIT;
}

/**
 * @brief OnlineRFClassifier::load
 * @param filename
 * @return
 */
bool OnlineRFClassifier::load(const char *filename)
{
    /* clear random forest */
    rf.clear();

    /* Load param files */
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!load_params(strfilename.c_str()))
        return false;

    /* Load model */
    TRAINED = false;
    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    rf.load(strfilename.c_str());
    TRAINED = rf.isTrained();
    return true;
}

/**
 * @brief OnlineRFClassifier::save
 * @param filename
 * @return
 */
bool OnlineRFClassifier::save(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_model_params");
    if(!INIT)
        return false;
    save_params(strfilename.c_str());

    strfilename = String(filename);
    strfilename.insert(strfilename.size()-4,"_model");
    if(!TRAINED)
        return false;
    rf.save(strfilename.c_str());

    return true;
}

/**
 * @brief OnlineRFClassifier::train
 * @param samples
 * @param classes
 * @return
 */
bool OnlineRFClassifier::train(Mat& samples, Mat& classes)
{
    bool success = false;
    double min, max;
    minMaxLoc(classes,&min,&max);
    if(min == max) // only one class
    {
        SINGLEVALUE = min;
    }
    else
    {
        cout << "samples size " << samples.size() << endl;
        cout << "Training Online Random Forest ..." << endl;
        success = rf.update(samples, classes);
        cout << "Done." << endl;

    }
    if(success)
        TRAINED = true;
    return success;
}

/**
 * @brief OnlineRFClassifier::predict
 * @param samples
 * @param useprobs
 * @return
 */
Mat OnlineRFClassifier::predict(Mat samples, bool useprobs)
{
    /* Make sure input data is floating points matrix */
    samples.convertTo(samples, CV_32F);

    /* Create output matrix */
    int nrows = samples.rows;
    int  ncols = 1;
    Mat predictedLabels = Mat::zeros(nrows, ncols, samples.type());
    if(TRAINED)
    {
        if(useprobs)
            predictedLabels = rf.predict_prob(samples);
        else
            predictedLabels = rf.predict(samples);
    }

    return predictedLabels;
}

/**
 * @brief OnlineRFClassifier::save_params
 * @param filename
 * @return
 */
bool OnlineRFClassifier::save_params(const char *filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if(!fs.isOpened())
        return false;
    if(!INIT)
        return false;

    /* Save parameters */
    fs << "priors" << Mat(vector<float>(rf.getParams().priors,rf.getParams().priors+10));
    fs << "max_depth" << rf.getParams().max_depth;
    fs << "min_sample_count" << rf.getParams().min_sample_count;
    fs << "regression_accuracy" << rf.getParams().regression_accuracy;
    fs << "use_surrogates" << rf.getParams().use_surrogates;
    fs << "max_categories" << rf.getParams().max_categories;
    fs << "calc_var_importance" << rf.getParams().calc_var_importance;
    fs << "nactive_vars" << rf.getParams().nactive_vars;
    fs << "max_num_of_trees_in_the_forest" << rf.getParams().term_crit.max_iter;
    fs << "forest_accuracy" << rf.getParams().term_crit.epsilon;
    fs << "termcrit_type" << rf.getParams().term_crit.type;
    fs << "nb_classes" << rf.getNbClasses();
    fs << "training_ratio" << rf.getTrainingRatio();
    fs << "nb_tree_per_update" << rf.getNbTreePerUpdate();
    fs << "last_updated_trees" << Mat(rf.getLastUpdated());
    fs << "online_saving" << rf.getOnlineSaving();
    fs.release();
    return true;
}

/**
 * @brief OnlineRFClassifier::load_params
 * @param filename
 * @return
 */
bool OnlineRFClassifier::load_params(const char *filename)
{
    INIT = false;
    if((filename != NULL) && (filename[0] == '\0'))
        return false;
    FileStorage fs(filename, FileStorage::READ);

    if(!fs.isOpened())
        return false;

    /* clear all */
    rf.clear();

    /* load parameters and init object */
    Mat priorsMat;
    fs["priors"] >> priorsMat;

    float* priors = new float[priorsMat.rows];
    for(int i = 0 ;  i < priorsMat.rows ; i++)
    {
        priors[i] = priorsMat.at<float>(i);
    }
    CvRTParams rf_params = CvRTParams((int) fs["max_depth"],
                             (int) fs["min_sample_count"],
                             (int) fs["regression_accuracy"],
                             (int) fs["use_surrogates"],
                             (int) fs["max_categories"],
                             priors,
                             (int) fs["calc_var_importance"],
                             (int) fs["nactive_vars"],
                             (int) fs["max_num_of_trees_in_the_forest"],
                             (float) fs["forest_accuracy"],
                             (int) fs["termcrit_type"]);
    rf.setNbClasses((int) fs["nb_classes"]);
    Mat last_updated_trees;
    fs["last_updated_trees"] >> last_updated_trees;
    rf.setLastUpdated(last_updated_trees);
    INIT = rf.init(rf_params,
            (float) fs["training_ratio"],
            (int) fs["nb_tree_per_update"],
            (int) fs["online_saving"]);
    fs.release();
    return true;
}
