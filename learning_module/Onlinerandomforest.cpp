/**
 * \file Onlinerandomforest.cpp
 * \brief OnlineRandomForest
 * \author Céline Craye
 * \version 0.1
 * \date 7 / 15 / 2015
 *
 * Please check header files for a detailed description
 *
 */

#include "Onlinerandomforest.h"
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common.h"

using namespace Common;
using namespace cv;


OnlineRandomForest::OnlineRandomForest()
{
    nb_trees = 0;
    nb_classes = 2; // by default, the classifier is binary
    INIT = false;
    TRAINED = false;
}

OnlineRandomForest::~OnlineRandomForest()
{
    for(int i = 0 ; i < nb_trees ; i++)
    {
        delete forest[i];
    }
}

bool OnlineRandomForest::init(CvRTParams params, float training_ratio, int nb_tree_per_update, bool online_saving)
{
    /* Initialize parameters */
    tree_params = CvDTreeParams( params.max_depth, params.min_sample_count,
        params.regression_accuracy, params.use_surrogates, params.max_categories,
        params.cv_folds, params.use_1se_rule, false, params.priors );
    forest_params = params;
    nb_trees = params.term_crit.max_iter;

    /* Initialize forest */
    forest.resize(nb_trees);
    for(size_t i = 0 ; i < forest.size() ; i++)
    {
        forest[i] = new CvDTree();
    }

    (training_ratio > 0 && training_ratio <= 1) ?
        this->training_ratio = training_ratio : this->training_ratio = 0.7;

    (nb_tree_per_update > 0  && nb_tree_per_update <= nb_trees) ?
        this->nb_trees_per_update = nb_tree_per_update : this->nb_trees_per_update = 1;

    this->ONLINE_SAVING = online_saving;

    INIT = true;
    TRAINED  =  false;
    return true;
}

bool OnlineRandomForest::clear()
{
    /* Reset params */
    tree_params = CvDTreeParams();
    forest_params = CvRTParams();

    /* Reset forest */
    for( int k = 0; k < nb_trees; k++ )
        delete forest[k];
    forest.clear();

    /* set default values */
    nb_trees = 0;
    nb_classes = 2; // by default, the classifier is binary
    last_updated_trees.clear();
    INIT = false;
    TRAINED = false;
    return true;
}

bool OnlineRandomForest::reset_learning()
{
    TRAINED = false;
    last_updated_trees.clear();
    for( int k = 0; k < nb_trees; k++ )
    {
        delete forest[k];
        forest[k] = new CvDTree();
    }
    return true;
}

/**
 * @brief OnlineRandomForest::train_all Train a random forest offline
 *        Data used for training won't be stored.
 * @param trainingData
 * @param trainingClasses
 * @return true if training was successful, false otherwise
 */
bool OnlineRandomForest::offline_train(Mat &trainingData, Mat & trainingClasses)
{
    assert(trainingData.rows == trainingClasses.rows);
    double min,max;
    minMaxLoc(trainingClasses,&min,&max);
    assert(min >= 0 && min < max);

    if(!INIT)
        return false;

    last_updated_trees.clear();
    /* Train each tree with a portion of the training set */
    for( int i = 0 ; i < nb_trees ; i++)
    {
        train_tree(trainingData,trainingClasses, i);

        last_updated_trees.push_back(i);
    }
    /* update nb_classes for evaluation */
    nb_classes = max + 1;
    TRAINED = true;
    return true;
}

/**
 * @brief OnlineRandomForest::update re-train a single tree with provided and previously stored data
 *        Data used for that training are stored for further updates.
 * @param trainingData
 * @param trainingClasses
 * @param treeIdx the tree index to be updated
 * @return true if training was successful, false otherwise
 */
bool OnlineRandomForest::update(Mat &trainingData, Mat &trainingClasses)
{
    assert(trainingData.rows == trainingClasses.rows);
    assert(!trainingClasses.empty());
    assert(trainingClasses.type() == CV_32F);

    double minClass,maxClass;
    minMaxLoc(trainingClasses,&minClass,&maxClass);
    assert(minClass >= 0);

    if(!INIT)
        return false;

    /* If model has not been trained yet, train as offline for initialization */
    if(!TRAINED)
        return offline_train(trainingData, trainingClasses);

    /* select random trees to re-train */
    vector<int> tree_idx;
    for (int i = 0 ; i < nb_trees; i++) tree_idx.push_back(i);
    std::random_shuffle ( tree_idx.begin(), tree_idx.end() );

    last_updated_trees.clear();

    /* train each tree */
    for(int i = 0 ; i < nb_trees_per_update ; i++)
    {
        train_tree(trainingData,trainingClasses, tree_idx[i]);
        last_updated_trees.push_back(tree_idx[i]);
    }

    return true;
}

/**
 * @brief OnlineRandomForest::predict predicts the classes of a set of samples
 * @param predictData
 * @return the predicted classes
 */
Mat OnlineRandomForest::predict(Mat& predictData)
{
    /* Initialize data */
    Mat predictValues = Mat::zeros(predictData.rows , 1, CV_32F); // output values
    CvDTreeNode* resultNode;

    if(TRAINED)
    {
        /* for each sample, predict the class */
        for(int j = 0 ; j < predictData.rows ; j++)
        {
            Mat predictHisto = Mat::zeros(1 , nb_classes, CV_8U); // output values
            int maxVal = 0;
            int maxClass = 0;
            /* The class is the most frequent among the forest */
            for(size_t i = 0 ; i < forest.size() ; i++)
            {
                resultNode = forest[i]->predict( predictData.row(j), Mat(), false);
                int classVal = (int)resultNode->value;
                predictHisto.at<uchar>(classVal) += 1;
                if(predictHisto.at<uchar>(classVal)>maxVal)
                {
                    maxVal = predictHisto.at<uchar>(classVal);
                    maxClass = classVal;
                }
            }
            predictValues.at<float>(j) = maxClass;
        }
    }
    return predictValues;
}

/**
 * @brief OnlineRandomForest::predict_prob Returns a fuzzy-predicted class label
 * The function works for binary classification problems only (labels 0 and 1).
 * It returns the number between 0 and 1.
 * @param predictData the sample matrix to evaluate
 * @return probability of class 1 for each sample
 */
Mat OnlineRandomForest::predict_prob(Mat &predictData)
{
    /* Initialize data */
    Mat predictValues = Mat::zeros(predictData.rows , 1, CV_32F); // output values
    CvDTreeNode* resultNode;
    if(nb_classes != 2)
    {
        cerr << "Cannot estimate probability for non binary problems" << endl;
        return predictValues;
    }

    if(TRAINED)
    {
        /* for each sample, predict the class */
        for(size_t i = 0 ; i < forest.size() ; i++)
        {
            for(int j = 0 ; j < predictData.rows ; j++)
            {
                resultNode = forest[i]->predict(predictData.row(j), Mat(), false);
                predictValues.at<float>(j) += (float)resultNode->value;
            }
        }
        /* Normalize to get probability */
        predictValues = predictValues/(float)nb_trees;
    }
    return predictValues;
}

/**
 * @brief OnlineRandomForest::load_tree load a single tree without data.
 * The tree model will be read-only
 * @param treeIdx
 * @param filename
 * @return true if loading was successful, false otherwise
 */
bool OnlineRandomForest::load_tree(int treeIdx, string filename)
{
    if(!INIT || treeIdx < -1 || treeIdx >= nb_trees)
        return false;

    forest[treeIdx]->load(filename.c_str());
    return true;
}

/**
 * @brief OnlineRandomForest::save_tree saves a single tree without data.
 * @param treeIdx
 * @param filename
 * @return true if saving was successful, false otherwise
 */
bool OnlineRandomForest::save_tree(int treeIdx, string filename)
{
    if(!INIT || treeIdx < -1 || treeIdx >= nb_trees)
        return false;

    forest[treeIdx]->save(filename.c_str());
    return true;
}

bool OnlineRandomForest::isInit()
{
    return INIT;
}

bool OnlineRandomForest::isTrained()
{
    return TRAINED;
}

CvRTParams OnlineRandomForest::getParams()
{
    return forest_params;
}

int OnlineRandomForest::getNbClasses()
{
    return nb_classes;
}

float OnlineRandomForest::getTrainingRatio()
{
    return training_ratio;
}

float OnlineRandomForest::getNbTreePerUpdate()
{
    return nb_trees_per_update;
}

vector<int> OnlineRandomForest::getLastUpdated()
{
    return last_updated_trees;
}

bool OnlineRandomForest::getOnlineSaving()
{
    return ONLINE_SAVING;
}

void OnlineRandomForest::setLastUpdated(Mat last_updated)
{
    last_updated_trees.clear();
    for(int i = 0 ; i < last_updated.rows ; i++)
    {
        last_updated_trees.push_back(last_updated.at<int>(i));
    }
}

void OnlineRandomForest::setNbClasses(int nb_classes)
{
    this->nb_classes = nb_classes;
}

bool OnlineRandomForest::train_tree(Mat &trainingData, Mat &trainingClasses, int treeIdx)
{
    /* Initialize shuffled vector indices */
    vector<int> indices_vect;
    for (int i = 0 ; i < trainingData.rows; i++) indices_vect.push_back(i);
    std::random_shuffle ( indices_vect.begin(), indices_vect.end() );

    vector<int> sample_idx = vector<int>(
                             indices_vect.begin(),
                             indices_vect.begin() + training_ratio*indices_vect.size());
    forest[treeIdx]->train(trainingData, CV_ROW_SAMPLE, trainingClasses,
                     Mat(), Mat(sample_idx), Mat(),Mat(), tree_params);
    return true;
}

/**
 * @brief OnlineRandomForest::load
 * @param filename
 * @return
 */
bool OnlineRandomForest::load(string filename)
{
    if(!INIT)
        return false;

    if(!ONLINE_SAVING || TRAINED) // load all model
    {
        last_updated_trees.clear();
        for(int i = 0 ; i < nb_trees ; i++)
            last_updated_trees.push_back(i);
    }
    else // else load only last updated models
    {
    }

    // for all trees, try to load model
    bool res = true;
    for(size_t i = 0 ; i < last_updated_trees.size() ; i++)
    {
        string treename = filename;
        string insertion = "/tree_" + to_string(last_updated_trees[i]);
        treename.insert(treename.size()-4,insertion);
        res = res & load_tree(last_updated_trees[i], treename);
    }
    TRAINED = res;
    return TRAINED;
}

bool OnlineRandomForest::save(string filename)
{
    if(!INIT)
        return false;

    if(!ONLINE_SAVING) // load all model
    {
        last_updated_trees.clear();
        for(int i = 0 ; i < nb_trees ; i++)
            last_updated_trees.push_back(i);
    }
    else // else load only last updated models
    {
    }

    /* Make directory */
    struct stat st = {0};
    string directoryname = filename;
    directoryname.replace(directoryname.size()-4,4, "");

    if (stat(directoryname.c_str(), &st) == -1) {
        mkdir(directoryname.c_str(), 0777);
    }

    // for all trees, try to save model
    bool res = true;
    for(size_t i = 0 ; i < last_updated_trees.size() ; i++)
    {
        string treename = filename;
        string insertion = "/tree_" + to_string(last_updated_trees[i]);
        treename.insert(treename.size()-4,insertion);
        res = res & save_tree(last_updated_trees[i], treename);
    }
    return res;

}



