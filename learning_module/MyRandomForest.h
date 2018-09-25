/**
 * \file MyRandomForest.h
 * \brief MyRandomForest
 * \author Céline Craye
 * \version 0.1
 * \date Feb 16, 2015
 *
 * this file is taken from OpenCV source code and intends to do the exact same thing as
 * CvRTrees. A few functions were added to the original class to provide measures such as
 * novelty, defined on Breiman's website. Those function are not used anymore in the code
 *
 */

#ifndef MYRANDOMFOREST_H_
#define MYRANDOMFOREST_H_

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

#include "opencv2/core/core.hpp"
#include <limits.h>

#include <map>
#include <string>
#include <iostream>
//#include <ml.h>
#include "precomp.hpp"

class MyRandomForest;
class MyForestTree;


struct MyForestTreeBestSplitFinder : cv::DTreeBestSplitFinder
{
    MyForestTreeBestSplitFinder() : cv::DTreeBestSplitFinder() {}
	virtual ~MyForestTreeBestSplitFinder();
	MyForestTreeBestSplitFinder( MyForestTree* _tree, CvDTreeNode* _node );
//	MyForestTreeBestSplitFinder( const MyForestTreeBestSplitFinder& finder, cv::Split );
    virtual void operator()(const cv::BlockedRange& range);
};

class MyForestTree: public CV_EXPORTS CvDTree
{
public:
	MyForestTree();
    virtual ~MyForestTree();

    virtual bool train( CvDTreeTrainData* trainData, const CvMat* _subsample_idx, MyRandomForest* forest );

    virtual int get_var_count() const {return data ? data->var_count : 0;}
    virtual void read( CvFileStorage* fs, CvFileNode* node, MyRandomForest* forest, CvDTreeTrainData* _data );

    /* dummy methods to avoid warnings: BEGIN */
    virtual bool train( const CvMat* trainData, int tflag,
                        const CvMat* responses, const CvMat* varIdx=0,
                        const CvMat* sampleIdx=0, const CvMat* varType=0,
                        const CvMat* missingDataMask=0,
                        CvDTreeParams params=CvDTreeParams() );

    virtual bool train( CvDTreeTrainData* trainData, const CvMat* _subsample_idx );
    virtual void read( CvFileStorage* fs, CvFileNode* node );
    virtual void read( CvFileStorage* fs, CvFileNode* node,
                       CvDTreeTrainData* data );
    /* dummy methods to avoid warnings: END */

protected:
    friend struct MyForestTreeBestSplitFinder;

    virtual CvDTreeSplit* find_best_split( CvDTreeNode* n );
    MyRandomForest* forest;
};



struct RTParams : public CV_EXPORTS_W_MAP CvDTreeParams
{
    //Parameters for the forest
    bool calc_var_importance; // true <=> RF processes variable importance
    int nactive_vars;
    CvTermCriteria term_crit;

    RTParams();
    RTParams( int max_depth, int min_sample_count,
                float regression_accuracy, bool use_surrogates,
                int max_categories, const float* priors, bool calc_var_importance,
                int nactive_vars, int max_num_of_trees_in_the_forest,
                float forest_accuracy, int termcrit_type );
};




class MyRandomForest : public CV_EXPORTS_W  CvStatModel
{
public:
	MyRandomForest();
    virtual ~MyRandomForest();
    virtual bool train( const CvMat* trainData, int tflag,
                        const CvMat* responses, const CvMat* varIdx=0,
                        const CvMat* sampleIdx=0, const CvMat* varType=0,
                        const CvMat* missingDataMask=0,
                        RTParams params=RTParams() );

    virtual bool train( CvMLData* data, RTParams params=RTParams() );
    virtual float predict( const CvMat* sample, const CvMat* missing = 0 ) const;
    virtual float predict_prob( const CvMat* sample, const CvMat* missing = 0 ) const;

    virtual bool train( const cv::Mat& trainData, int tflag,
                       const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
                       const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(),
                       const cv::Mat& missingDataMask=cv::Mat(),
                       RTParams params=RTParams() );
    virtual float predict( const cv::Mat& sample, const cv::Mat& missing = cv::Mat() ) const;
    virtual float predict_prob( const cv::Mat& sample, const cv::Mat& missing = cv::Mat() ) const;
    virtual cv::Mat getVarImportance();

    virtual void clear();

    virtual const CvMat* get_var_importance();
    virtual float get_proximity( const CvMat* sample1, const CvMat* sample2,
        const CvMat* missing1 = 0, const CvMat* missing2 = 0 ) const;

    virtual float calc_error( CvMLData* data, int type , std::vector<float>* resp = 0 ); // type in {CV_TRAIN_ERROR, CV_TEST_ERROR}

    virtual float get_train_error();

    virtual void read( CvFileStorage* fs, CvFileNode* node );
    virtual void write( CvFileStorage* fs, const char* name ) const;

    CvMat* get_active_var_mask();
    CvRNG* get_rng();

    int get_tree_count() const;
    MyForestTree* get_tree(int i) const;

    // new functions
    bool makeClusters(const cv::Mat trainData, int K);
    bool build_node_map(const cv::Mat& trainData);
    std::vector<std::vector<int> > getClusters(); //get a list of clusters containing indices of samples from training set
    int getNearestNeighbor(const CvMat* sample);
    double getNovelty(const CvMat* sample); // provides a measure of novelty
    double getOutlierScore(const CvMat* sample);
    double getBinUncertainty(const CvMat* sample);
    int getCluster(const CvMat* sample);
    void displayRF();


protected:
    virtual std::string getName() const;

    virtual bool grow_forest( const CvTermCriteria term_crit );

    // array of the trees of the forest
    MyForestTree** trees;
    CvDTreeTrainData* data;
    int ntrees;
    int nclasses;
    double oob_error;
    CvMat* var_importance;
    int nsamples;

    cv::RNG* rng;
    CvMat* active_var_mask;

    // new variables
    bool HAS_NODE_MAP; // you can do it
    bool HAS_CLUSTER_NODE_MAP; // you can do it
    int NB_CLUSTERS;
    std::map<CvDTreeNode*,std::vector<int> > node_map; // maps terminal nodes with indices of the training set.
    int trainDataSize; // number of samples in the training set.
    std::map<CvDTreeNode*,std::vector<int> > node_map_clusters; // maps terminal nodes with cluster indices.
};


#endif /* MYRANDOMFOREST_H_ */
