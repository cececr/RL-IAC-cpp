/**
 * \file LearningM.cpp
 * \brief LearningM
 * \author Céline Craye
 * \version 0.1
 * \date Oct 13, 2014
 *
 * Please refere to header file for a detailed description
 *
 */


#include "LearningM.h"

#include <dirent.h>
#include <errno.h>

#include "printdebug.h"
#define USE_RTREE

using namespace cv;
/**
 * @brief LearningM::LearningM constructor
 * @param max_samples max number of samples in the dataset
 * @param make_balanced_dataset make a 50-50 balanced dataset
 * @param use_random_replace delete exceeding samples randomly
 */
LearningM::LearningM(int max_samples,
                     bool make_balanced_dataset,
                     bool use_random_replace)
{
    classif = 0;
    classifierType = -1;
    this->max_samples = max_samples;
    this->make_balanced_dataset = make_balanced_dataset;
    this->use_random_replace = use_random_replace;
}

/**
 * @brief LearningM::~LearningM  destructor
 */
LearningM::~LearningM() {
    if(classif !=0)
    {
        delete classif;
        classif = 0;
    }
}


/**
 * @brief LearningM::load
 *        load the model stored at the specified location
 * @param filename path to the file storing the model
 * @return true if loading was successful, false otherwise
 */
bool LearningM::load(const char *filename)
{
    bool res;
    res = loadParams(filename);
    res = loadDataset(filename);
    res = loadModel(filename);

    return res;
}


/**
 * @brief LearningM::loadModel
 *        load the classifier model only
 * @param filename path to the file storing the model
 * @return true if loading was successful, false otherwise
 */
bool LearningM::loadModel(const char *filename)
{
    if(classif != 0)
        return classif->load(filename);
    return false;
}

/**
 * @brief LearningM::loadParams
 *        load the classifier parameters only
 * @param filename path to the file storing the parameters
 * @return true if loading was successful, false otherwise
 */
bool LearningM::loadParams(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_params");
    FileStorage fs(strfilename.c_str(), FileStorage::READ);
    if( !fs.isOpened())
        return false;
//    fs["max_samples"] >> max_samples;
//    fs["make_balanced_dataset"] >> make_balanced_dataset;
//    fs["use_random_replace"] >> use_random_replace;
//    fs["classifierType"] >> classifierType;


    /* Init classifier */
    init((int)fs["classifierType"],
         (int)fs["max_samples"],
         (int)fs["make_balanced_dataset"],
         (int)fs["use_random_replace"],
         strfilename);
    fs.release();

    return true;
}

/**
 * @brief LearningM::loadDataset
 *        load the classifier dataset only
 * @param filename path to the file storing the dataset
 * @return true if loading was successful, false otherwise
 */
bool LearningM::loadDataset(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_trainingSet");
    FileStorage fs(strfilename.c_str(), FileStorage::READ);
    if(!fs.isOpened())
        return false;

    fs["train_data"] >> trainingData ;
    fs["train_classes"] >> trainingClasses ;
    fs.release();
    return true;
}

/**
 * @brief LearningM::save
 *        save current model in a file
 * @param filename path to the file we want the model to be saved
 * @return true if saving was successful, false otherwise
 */
bool LearningM::save(const char *filename)
{

    bool res = saveModel(filename);
    res = saveDataset(filename);
    res = saveParams(filename);
    return res;
}

/**
 * @brief LearningM::saveModel
 *        save the classifier model only
 * @param filename path to the file we want the model to be saved
 * @return true if savinf was successful, false otherwise
 */
bool LearningM::saveModel(const char *filename)
{
    if(classif != 0)
        return classif->save(filename);
    return false;
}

/**
 * @brief LearningM::saveDataset
 *        save the classifier dataset only
 * @param filename path to the file we want the dataset to be saved
 * @return true if savinf was successful, false otherwise
 */
bool LearningM::saveDataset(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_trainingSet");
    FileStorage fs(strfilename.c_str(), FileStorage::WRITE);
    if( !fs.isOpened())
        return false;

    fs << "train_data" << trainingData;
    fs << "train_classes" << trainingClasses;
    fs.release();
    return true;
}

/**
 * @brief LearningM::saveParams
 *        save the model parameters only
 * @param filename path to the file we want the dataset to be saved
 * @return true if savinf was successful, false otherwise
 */
bool LearningM::saveParams(const char *filename)
{
    string strfilename(filename);
    strfilename.insert(strfilename.size()-4,"_params");
    FileStorage fs(strfilename.c_str(), FileStorage::WRITE);

    fs << "max_samples" << max_samples;
    fs << "make_balanced_dataset" << make_balanced_dataset;
    fs << "use_random_replace" << use_random_replace;
    fs << "classifierType" << classifierType;

    fs.release();
    return true;
}

/**
 * @brief LearningM::isInit
 * @return true if the classifier is initialized, false otherwise
 */
bool LearningM::isInit()
{
	if(classif == 0)
		return false;
	return classif->isInit();
}

/**
 * @brief LearningM::isTrained
 * @return true if the classifier is trained, false otherwise
 */
bool LearningM::isTrained()
{
	if(classif == 0)
		return false;
    return classif->isTrained();
}

/**
 * @brief LearningM::getClassifierType
 * @return the index type of the used classifier
 */
int LearningM::getClassifierType()
{
    return classifierType;
}

/**
 * @brief LearningM::getClassifier
 * @return a pointer to the current classifier
 */
Classifier *LearningM::getClassifier()
{
    return classif;
}

/**
 * @brief LearningM::getTrainingData
 * @return the training samples
 */
Mat LearningM::getTrainingData()
{
    return trainingData;
}

/**
 * @brief LearningM::init initialize classifier
 * @param classifier The classifier type you want to use. LearningM::My_RANDOM_TREE is recommended
 * @param max_samples  max number of samples in the dataset. 0 means infinity
 * @param make_balanced_dataset  make a 50-50 balanced dataset. Default is false
 * @param use_random_replace delete exceeding samples randomly. Default is false
 * @param param_file path to a file containing the parameters of the classifier.
 *        Should be the same format as the one provided by saveParams() function
 * @return wheter initialization was successful or not
 */
bool LearningM::init(int classifier,
                     int max_samples,
                     bool make_balanced_dataset,
                     bool use_random_replace,
                     string param_file)
{
	bool res = false;
    this->max_samples = max_samples;
    this->make_balanced_dataset = make_balanced_dataset;
    this->use_random_replace = use_random_replace;
	classifierType = classifier;
	if(classifier == My_RANDOM_TREE)
	{
		classif = new MyRFClassifier();
	}
	else if(classifier == RANDOM_TREE)
	{
		classif = new RFClassifier();
	}
	else if(classifier == NEURAL_NET)
	{
		classif = new MLPClassifier();
	}
    else if(classifier == ONLINE_RANDOM_TREE)
    {
        classif = new OnlineRFClassifier();
    }
	else if(classifier == SVM)
	{
		return false;
	}
	else
		return false;

    res = classif->init(param_file);
    return res;
}

/**
 * @brief LearningM::initTrainingSet
 *        initialize training set with a batch of training data
 * @param trainingData the training samples
 * @param trainingClasses the training labels
 * @return true if provided data is compatible with initialization
 */
bool LearningM::initTrainingSet(Mat trainingData, Mat trainingClasses)
{
    if(trainingClasses.empty() && trainingData.empty())
        return false;

    if(trainingData.rows != trainingClasses.rows)
        return false;

    assert(trainingData.rows == trainingClasses.rows);
    assert(trainingData.type() == CV_32F);
    assert(trainingClasses.type() == CV_32F);

    this->trainingClasses = trainingClasses;
    this->trainingData = trainingData;
    return true;
}

/**
 * @brief LearningM::reset
 *         empty training set and start a new classification from scratch
 * @return true anyways
 */
bool LearningM::reset()
{
	trainingData.release();
	trainingClasses.release();
	if(classif != 0)
		delete classif;
	return true;
}

/**
 * @brief LearningM::addBatchSamples
 *        add a set of training sample and labels to the current dataset.
 *        filters dataset to balance the data or remove sample depending on the provided options
 * @param samples the new set of training samples
 * @param labels the new set of training labels
 * @return true if data was successfuly updated
 */
bool LearningM::addBatchSamples(Mat samples, Mat labels)
{
    assert(samples.rows == labels.rows);
    assert(samples.type() == CV_32F);
    assert(labels.type() == CV_32F);
    trainingData.push_back(samples);
    trainingClasses.push_back(labels);

    /* If balanced dataset is required, remove samples */
    if(make_balanced_dataset)
    {
        int nzero_ct = countNonZero(trainingClasses);
        Mat mask;
        if(nzero_ct > trainingClasses.rows/2)
        {
            inRange(trainingClasses,1,1,mask);
        }
        else
        {
            inRange(trainingClasses,0,0,mask);
        }
        int new_nrows = 2*min(nzero_ct, trainingClasses.rows-nzero_ct);
        Mat newTrainingData = Mat::zeros(new_nrows, trainingData.cols, CV_32F);
        Mat newTrainingClasses = Mat::zeros(new_nrows, trainingClasses.cols, CV_32F);
        int i = 0, k = 0;
        for(int j = 0 ; j < trainingClasses.rows ; j++)
        {
            if(k < new_nrows && mask.at<uchar>(trainingClasses.rows-j-1) == 0)
            {
                trainingData.row(trainingClasses.rows-1-j).copyTo(newTrainingData.row(k));
                trainingClasses.row(trainingClasses.rows-1-j).copyTo(newTrainingClasses.row(k));
                k++;
            }
            else
            {
                if( k < new_nrows && i < new_nrows/2)
                {
                    trainingData.row(trainingClasses.rows-1-j).copyTo(newTrainingData.row(k));
                    trainingClasses.row(trainingClasses.rows-1-j).copyTo(newTrainingClasses.row(k));
                    i++;
                    k++;
                }
            }
        }
        trainingData = newTrainingData;
        trainingClasses = newTrainingClasses;
    }
    if(max_samples > 0)
    {
        if(trainingClasses.rows > max_samples)
        {
            if(use_random_replace)
            {
                // shuffle dataset
                Mat trainingDataCopy = trainingData.clone();
                Mat trainingClassesCopy = trainingClasses.clone();
                vector<int> rand_idx;
                for(int i = 0 ; i < trainingClasses.rows ; i++)
                    rand_idx.push_back(i);
                std::random_shuffle(rand_idx.begin(),rand_idx.end());
                for(int i = 0 ; i < trainingClasses.rows ; i++)
                {
                    trainingData.row(i) = trainingDataCopy.row(rand_idx[i]);
                    trainingClasses.row(i) = trainingClassesCopy.row(rand_idx[i]);
                }
            }
            // remove first samples
            int nsamplesRemoved = trainingClasses.rows - max_samples;
            trainingData(Rect(0,
                              nsamplesRemoved,
                              trainingData.cols,
                              trainingData.rows - nsamplesRemoved)).copyTo(trainingData);
            trainingClasses(Rect(0,
                                 nsamplesRemoved,
                                 trainingClasses.cols,
                                 trainingClasses.rows - nsamplesRemoved)).copyTo(trainingClasses);
        }
    }
    return true;
}

/**
 * @brief LearningM::addSingleSample
 *        add a single pair of training sample and label to the current dataset.
 * @param sample the new training sample
 * @param label the new training label
 * @return true anyways
 */
bool LearningM::addSingleSample(Mat sample, Mat label)
{
    assert(sample.rows == 1 && label.rows == 1);
    trainingData.push_back(sample);
    trainingClasses.push_back(label);
    return true;
}

/**
 * @brief LearningM::train
 *        train the classifier with accumulated samples
 * @return true if training was successful
 */
bool LearningM::train()
{
	/* Make sure input data is floating points matrices */
	trainingData.convertTo(trainingData, CV_32F);
	trainingClasses.convertTo(trainingClasses, CV_32F);
    if(isInit() && !trainingData.empty() && !trainingClasses.empty())
		return classif->train(trainingData,trainingClasses);
	else return false;
}


/**
 * @brief LearningM::predict
 *        Predict class of a set of samples
 * @param samples The matrix of samples. Each row is a sample.
 * @param useprobs Whether you want probabilistic or hard decision
 * @return A matrix of estimates
 */
Mat LearningM::predict(Mat samples, bool useprobs)
{
	Mat predictedLabels = classif->predict(samples, useprobs);
	return predictedLabels;
}


/**
 * @brief LearningM::estimateSaliencyFromFeatureMap
 *        Estimate saliency directly from a vector of feature maps
 * @param featureMaps the map of features
 * @param subsampling_fact downsampling factor, how many pixels do we skip for estimation (2 or 4 is ok)
 * @param enhance_display_factor correction that makes display look nicer. (use between 0 and 3)
 * @return
 */
Mat LearningM::estimateSaliencyFromFeatureMap(vector<Mat> & featureMaps,
                                              int subsampling_fact,
                                              int enhance_display_factor)
{
    assert(featureMaps.size() > 0);
    Size newSize(featureMaps[0].cols/subsampling_fact,featureMaps[0].rows/subsampling_fact);

    Mat result  = Mat::zeros(newSize.height, newSize.width, CV_8U);
	if(isTrained())
	{
		/* Convert feature map to appropriate format */
        Mat matSamples;
        merge(featureMaps,matSamples);
        resize(matSamples,matSamples,newSize,0,0,INTER_NEAREST);
        matSamples = matSamples.reshape(1,matSamples.cols*matSamples.rows);
		/* Predict and reconstruct */
		Mat predictedLabels = classif->predict(matSamples, true);
        result = predictedLabels.reshape(1,newSize.height);

        /* if enhancement factor, process */
        /* (just makes nicer display with stronger difference between salient and not salient) */
        for(int i = 0 ;  i < enhance_display_factor ; i++)
        {
            result = result.mul(result);
        }
	}
    /* Reformat */
    resize(result,result, featureMaps[0].size(), INTER_LINEAR);
    result = result * 255;
    result.convertTo(result, CV_8U);
    return result;
}

cv::Point sampleIdx2pixel(int idx, Size sampleSize, Size outputSize)
{
    int row_idx = idx/sampleSize.width;
    int col_idx = idx % sampleSize.height;
    return cv::Point(col_idx*(float)outputSize.width/(float)sampleSize.width,
                     row_idx*(float)outputSize.height/(float)sampleSize.height);
}
int pixel2sampleIdx(cv::Point pixel, Size sampleSize)
{
    int row_idx = pixel.y/sampleSize.height;
    int col_idx = pixel.x/sampleSize.width;
    return row_idx*sampleSize.width+col_idx;
}

cv::Mat fill_result_superpixel(cv::Mat superpixels, cv::Mat labels)
{
    /* Get number of superpixels */
    double min, max;
    minMaxLoc(superpixels,&min, &max);
    int nb_superpixels = max + 1;
    float lx = labels.cols;
    float ly = labels.rows;
    float sx = superpixels.cols;
    float sy = superpixels.rows;
    labels = labels.reshape(1,lx*ly);
    superpixels.convertTo(superpixels, CV_32S);
    cv::Mat result_map = cv::Mat::zeros(superpixels.size(), CV_32F);
    cv::Mat histogram_superpix_labels = cv::Mat::zeros(nb_superpixels,labels.rows*labels.cols,CV_32F);
    for(int x = 0 ; x < superpixels.cols ; x++)
    {
        for(int y = 0 ; y < superpixels.rows ; y++)
        {
            int superpixel_idx = superpixels.at<int>(y,x);
            int label_idx = floor((float)y/sy*ly)*lx+floor((float)x/sx*lx);
            histogram_superpix_labels.at<float>(superpixel_idx,label_idx) += 1;
        }
    }
    vector<float> superpixel_values;
    for(int i = 0 ; i < nb_superpixels ; i++)
    {
        /* using histogram-weighted label values */
        cv::Mat row_mat = histogram_superpix_labels.row(i).clone();
        Mat prod = row_mat*labels;
        superpixel_values.push_back( prod.at<float>(0) / sum(histogram_superpix_labels.row(i)).val[0]);

        /* using mode value of the superpixel */
//        double min, max;
//        cv::minMaxLoc(histogram_superpix_labels.row(i),&min,&max);
//        superpixel_values.push_back(labels.at<float>(max));
    }
    for(int x = 0 ; x < superpixels.cols ; x++)
    {
        for(int y = 0 ; y < superpixels.rows ; y++)
        {
            int superpixel_idx = superpixels.at<int>(y,x);
            result_map.at<float>(y,x) = superpixel_values[superpixel_idx];
        }
    }
    return result_map;
}

Mat LearningM::estimateSaliencyFromFeatureMap(vector<Mat> &featureMaps, Mat superpixels, int enhance_display_factor)
{
    assert(featureMaps.size() > 0);
    Size newSize = superpixels.size();

    Mat result  = Mat::zeros(newSize.height, newSize.width, CV_32F);
    if(isTrained())
    {
        /* Convert feature map to appropriate format */
        Mat matSamples;
        merge(featureMaps,matSamples);
        Size sampleSize = matSamples.size();
        matSamples = matSamples.reshape(1,matSamples.cols*matSamples.rows);
        /* Predict and reconstruct */
        clock_t t = clock();
        Mat predictedLabels = classif->predict(matSamples, true);
        PrintDebug::getTimeDiff(t,"prediction time");
        predictedLabels = predictedLabels.reshape(1,sampleSize.height);

        /* Replace superpixels by their mode values */
        double min, max;
        minMaxLoc(superpixels,&min, &max);
//        superpixels.convertTo(superpixels,CV_8U);
        result = fill_result_superpixel(superpixels,predictedLabels);
        PrintDebug::getTimeDiff(t,"fill superpixels");


        /* if enhancement factor, process */
        /* (just makes nicer display with stronger difference between salient and not salient) */
        for(int i = 0 ;  i < enhance_display_factor ; i++)
        {
            result = result.mul(result);
        }
    }
    /* Reformat */
    result = result * 255;
    result.convertTo(result, CV_8U);
    cout << result.size() << endl;
    return result;
}
