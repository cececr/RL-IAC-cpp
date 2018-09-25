/**
 * \file saliencylearningexperiment.cpp
 * \brief SaliencyLearningExperiment
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 9 / 2015
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#include <highgui.h>

#include "saliencylearningexperiment.h"
#include "bottom_up/saliencyDetectionBMS.h"
#include "bottom_up/saliencyDetectionHou.h"
#include "bottom_up/saliencyDetectionItti.h"
#include "bottom_up/saliencyDetectionRudinac.h"
#include "bottom_up/VOCUS2.h"
#include "feature_extractor/deepfeatureextractor.h"

using namespace std;
using namespace cv;

SaliencyLearningExperiment::SaliencyLearningExperiment(std::string param_file)
{
    /* init values */
    params_filename = param_file;
    bottom_up_extractor = 0;
    feature_extractor = 0;
    time = 0;
    /* Load params */
    SaliencyFileParser fp;

    fp.parse_param_file(param_file,global_params);
    if(global_params.string["outputLogFile"].empty())
    {
        std::string logname = param_file;
        logname.replace(param_file.length()-4,4,"_log.txt");
        global_params.string["outputLogFile"] = logname;
    }
    cout << global_params.string["outputLogFile"] << endl;
    init_missing_params();
    return;
}

SaliencyLearningExperiment::~SaliencyLearningExperiment()
{
    if(feature_extractor != 0)
        delete feature_extractor;
    if(bottom_up_extractor != 0)
        delete bottom_up_extractor;
}

void SaliencyLearningExperiment::run()
{
    /*Init objects*/
    init_environment();
    init_feature_extractor();
    init_segmenter();
    init_learner();
    init_action_selection();
    init_evaluation();
    init_experiment();

//    /* Evaluation on a single frame */
//    std::vector<string> names;
//    names.push_back("/home/thales/Matlab/IJRR/Mosaic/ICRA/rgb/rosbag_1_184.png");
//    names.push_back("/home/thales/Matlab/IJRR/Mosaic/ICRA/rgb/rosbag_1_304.png");
//    names.push_back("/home/thales/Matlab/IJRR/Mosaic/ICRA/rgb/rosbag_1_335.png");
//    names.push_back("/home/thales/Matlab/IJRR/Mosaic/ICRA/rgb/rosbag_1_352.png");
//    names.push_back("/home/thales/Matlab/IJRR/Mosaic/ICRA/rgb/rosbag_1_260.png");
//    std::vector<Mat> ims;
//    for(int i = 0 ; i < names.size() ; i++)
//    {
//        cv::Mat im = imread(names[i]);
//        ims.push_back(im);
//    }

    clock_t t;
    for(int i = 0 ; i < nb_steps ; i++)
    {
        t = clock();
        evaluate();
        PrintDebug::getTimeDiff(t,"evaluate");
        t = clock();
        take_action(i);
        PrintDebug::getTimeDiff(t,"take action");
        if(learn)
        {
            t = clock();
            get_input();
            PrintDebug::getTimeDiff(t,"get input");
            t = clock();
            extract_features();
            PrintDebug::getTimeDiff(t,"extract features");
            t = clock();
            segment_objects();
            PrintDebug::getTimeDiff(t,"segment objects");
            t = clock();
            get_saliency_map();
            PrintDebug::getTimeDiff(t,"get sal map");
            t = clock();
            display();
            PrintDebug::getTimeDiff(t,"display");
            t = clock();
            feed_learners();
            PrintDebug::getTimeDiff(t,"feed learners");
            t = clock();
            train_learners();
            PrintDebug::getTimeDiff(t,"train learners");
//            save_saliency_map();
        }
//        /* Evaluation on a single frame */
//        for(int k = 0 ; k < ims.size() ; k++)
//        {
//        std::vector<cv::Mat> im_feature_map = feature_extractor->getFeatureMap(ims[k]);
//        cv::Mat im_superpixels = feature_extractor->getSuperpixelsMap(ims[k]);
//        cv::Mat im_saliency_map = regions_learner.get_saliency_map_at(im_feature_map,im_superpixels, 0);
//        ostringstream ss;
//        ss << i;
//        string str = names[k] + ss.str() + ".png";
//        imwrite(str, im_saliency_map);
//        }

    }
    save_model();
}

void SaliencyLearningExperiment::run_bottom_up()
{
    /*Init objects*/
    init_environment();
    init_bottom_up();
    init_evaluation();
    for(int i = 0 ; i < environment.get_inputs_size() ; i++)
    {
        evaluate();
        current_position = i;
        get_input();
        saliency_map = bottom_up_extractor->getSalMap(rgb_input);
        display();
        save_saliency_map();
    }
}

void SaliencyLearningExperiment::run_segmentation_only()
{
    init_environment();
    init_segmenter();
    global_params.num["useSegImages"] = 0;
    string str = global_params.string["outputSaliencyDir"] + "/boundingboxes.txt";
    ofstream* logfile = new ofstream(str.c_str() , ios::out | ios::trunc);
    for(int i = 0 ; i < environment.get_inputs_size() ; i++)
    {
        current_position = i;
        *logfile << environment.get_input_name(current_position) << endl;
        get_input();
        segment_objects();
        display();
        /* seg map plays the role of sal map here */
        saliency_map = segmentation_mask;
        save_saliency_map();
        std::vector<Rect> boxes = object_segmenter.Get2DBBox();
        for (int k = 0 ; k < boxes.size() ; k++)
        {
            *logfile << boxes[k].x << " "
                     << boxes[k].y << " "
                     << boxes[k].width << " "
                     << boxes[k].height << endl;
        }
    }
    logfile->close();
}

void SaliencyLearningExperiment::run_offline_learning()
{
    /*Init objects*/
    init_environment();
    init_feature_extractor();
    init_segmenter();
    init_learner();
    for(int i = 0 ; i < environment.get_inputs_size() ; i++)
    {
        current_position = i;
        get_input();
        extract_features();
        segment_objects();
        feed_learners();
        display();
    }
    /* train all and save models */
    regions_learner.train();
    save_model();
}

void SaliencyLearningExperiment::run_offline_saliency()
{
    /*Init objects*/
    init_environment();
    init_feature_extractor();
    init_learner();
    init_evaluation();
    /* load models */
    load_model();
    for(int i = 0 ; i < environment.get_inputs_size() ; i++)
    {
        evaluate();
        current_position = i;
        get_input();
        extract_features();
        get_saliency_map();
        display();
        save_saliency_map();
    }
}

bool SaliencyLearningExperiment::take_action(int i)
{

    int current_region = environment.get_region(current_position);
    region_states = regions_learner.get_regions_progress();

    int position, time_to_pos;
    cout << "region " << current_region << endl;
    if(i < 500)
    {
        action_selector.minval = -1;
        action_selector.maxval = 436;
    }
    else
    {

        action_selector.minval = 436;
        action_selector.maxval = 10000;
    }
    action_selector.select_position(current_region, region_states, &position, &time_to_pos);
    cout << "position " << position << endl;

    if(position == ActionSelectionM::LEARN)
    {
        learn = true;
        time += time_to_pos;
    }
    else
    {
        current_position = position;
        time += time_to_pos;
        learn = false;
    }
    return true;
}

bool SaliencyLearningExperiment::get_input()
{
    environment.get_inputs(current_position,rgb_input,depth_input);
    return true;
}

bool SaliencyLearningExperiment::extract_features()
{
    feature_map = feature_extractor->getFeatureMap(rgb_input);
    return true;
}

bool SaliencyLearningExperiment::segment_objects()
{
    /* Case where segmentation map is already provided */
    if( global_params.num["useSegImages"] == 1)
    {
        environment.get_segmentation_input(current_position,segmentation_mask);
    }
    /* Otherwise, call segmenter */
    else
    {
        /* If use floor tracker */
        if( global_params.num["useFloorTracker"] == 1)
        {
            floor_tracker.update_frame(rgb_input, depth_input);
            floor_tracker.track();
            object_segmenter.SetFloorEquation(floor_tracker.get_ptr_floor_coeff());
        }
        object_segmenter.setInputCloudFromDepth(depth_input);
        if(!object_segmenter.hasFloorEquation())
        {
            object_segmenter.EstimateMainPlaneCoefs();
        }
        object_segmenter.Segment();
        segmentation_mask = object_segmenter.GetSegmentationMask();
    }
    return true;
}

bool SaliencyLearningExperiment::feed_learners()
{
    int current_region = environment.get_region(current_position);
    regions_learner.add_image_samples(current_region,feature_map, segmentation_mask);
    return true;
}

bool SaliencyLearningExperiment::train_learners()
{
    int current_region = environment.get_region(current_position);
    regions_learner.train(current_region);
    return true;
}

bool SaliencyLearningExperiment::evaluate()
{
    int current_region = environment.get_region(current_position);
    if(!bottom_up_extractor)
        experiment_eval.update_log_file(current_region, &regions_learner, time);
    else
        experiment_eval.update_log_file(current_region);
    return true;
}

bool SaliencyLearningExperiment::get_saliency_map()
{
    int current_region = environment.get_region(current_position);
    if(global_params.num["useSuperpixels"]== 1)
    {
        clock_t t;
        t = clock();
        cv::Mat superpixels = feature_extractor->getSuperpixelsMap(rgb_input);
        PrintDebug::getTimeDiff(t, "superpixels");
        saliency_map = regions_learner.get_saliency_map_at(feature_map,superpixels, current_region);
    }
    else
    {
        saliency_map = regions_learner.get_saliency_map_at(feature_map,current_region);
    }

    if(global_params.num["useFusedMap"]== 1)
    {
        fused_map = saliency_map;
        cv::Mat seg_salient, seg_not_salient;
        inRange(segmentation_mask,0,0,seg_not_salient);
        inRange(segmentation_mask,255,255,seg_salient);
        fused_map.setTo(0, seg_not_salient);
        fused_map.setTo(255, seg_salient);
    }
    return true;
}

bool SaliencyLearningExperiment::display()
{
    int current_region = environment.get_region(current_position);

    cout << environment.get_input_name(current_position) << endl;
    imwrite(environment.get_input_name(current_position), saliency_map);
    if(!saliency_map.empty()
       && global_params.string["displayFrames"].find("sal") != std::string::npos)
    {
        resize(saliency_map, saliency_map, rgb_input.size());
        imshow("saliency map", saliency_map);

        if(global_params.num["useFusedMap"] == 1)
        {
            imshow("fused map", fused_map);
        }
    }
    if(!rgb_input.empty()
       && global_params.string["displayFrames"].find("rgb") != std::string::npos)
    {
        imshow("RGB input", rgb_input);
    }
    if(!segmentation_mask.empty()
       && global_params.string["displayFrames"].find("seg") != std::string::npos)
    {
        imshow("segmentation", segmentation_mask);
    }
    if(global_params.num["displayRegionState"] == 1)
    {
        cv::Mat internal_state = regions_learner.draw_internal_state(current_region);
        if(!internal_state.empty())
        {
            imshow("internal state", internal_state);
        }
    }
    if(global_params.num["displayWorldMap"] == 1)
    {
       cv::Mat world_map = environment.draw_world_map(current_region, region_states);
       if(!world_map.empty())
       {
           imshow("world map", world_map);
       }
    }
    waitKey(10);
    return true;
}

bool SaliencyLearningExperiment::save_model()
{
    return regions_learner.save_all_learners(global_params.string["modelName"]);
}

bool SaliencyLearningExperiment::load_model()
{
    if(!regions_learner.load_all_learners(global_params.string["modelName"]))
    {
        cerr << "load_model: Could not load all models properly" << endl;
        return false;
    }
    return true;
}

/**
 * @brief int2string converts an int to a string, filling with 0s
 * @param i the int to convert
 * @param fill_val number of characters in the string
 * @return
 */
std::string int2string(int i, int fill_val)
{
    std::ostringstream ss;
    ss << std::setw( fill_val ) << std::setfill( '0' ) << i;
    return ss.str();
}

bool SaliencyLearningExperiment::save_saliency_map()
{
    if(saliency_map.empty())
        return false;

    string output_filename = environment.get_input_name(current_position) ;
    if(!global_params.string["outputSaliencyDir"].empty())
    {
//        time ++;
        output_filename = global_params.string["outputSaliencyDir"] + "/"
                        + environment.get_input_name(current_position);
//                        + "_" + int2string(time,5) + ".png";
    }
    cout << output_filename << endl;
    // saliency map
    if(!cv::imwrite(output_filename, saliency_map))
    {
        cerr << "save_saliency_map: Could not save saliency map into "
             << output_filename << endl;
        return false;
    }
//    // internal state
//    Mat internal_state = regions_learner.draw_internal_state(environment.get_region(current_position));
//    output_filename = output_filename + "internal_state.png";
//    cv::imwrite(output_filename, internal_state);

    return true;
}

bool SaliencyLearningExperiment::init_environment()
{
    SaliencyFileParser fp;
    std::vector<std::string> inputs_list;
    std::map<std::string, int> input_region_map;
    WorldMapGraph::GraphStruct graph_struct;

    /* try to load data from depth map names */
    fp.get_dir_file_names(global_params.string["inputPath"], inputs_list, "_depth", true);
    /* if failed, try to load data from GT map names */
    if(inputs_list.empty())
    {
        fp.get_dir_file_names(global_params.string["inputPath"], inputs_list, "_GT", true);
    }
    if(inputs_list.empty())
    {
        cerr << "No input data found in " << global_params.string["inputPath"] << endl;
    }


    if(!global_params.string["inputRegionPath"].empty())
    {
        fp.parse_input_region_file(global_params.string["inputRegionPath"], input_region_map);
    }
    if(!global_params.string["regionMapPath"].empty())
    {
        fp.parse_region_map_file(global_params.string["regionMapPath"], graph_struct);
    }
    environment.init(global_params.string["inputPath"],inputs_list,input_region_map,graph_struct);
    environment.init_world_map(0);
    return true;
}

bool SaliencyLearningExperiment::init_experiment()
{
    nb_steps = global_params.num["nbSteps"];
    if(nb_steps < 0 )
        nb_steps = 0;
    initial_position = global_params.num["initialPosition"];
    if(initial_position > environment.get_inputs_size() || initial_position < 0)
        initial_position = 0;
    current_position = initial_position;
    return true;
}

bool SaliencyLearningExperiment::init_feature_extractor()
{

    string featureParamsFile = global_params.string["featureParamFile"];
    SaliencyFileParser::Param_Struct init_params;
    init_params.env_variables = global_params.env_variables;
    SaliencyFileParser fp;
    fp.parse_param_file(featureParamsFile, init_params);

    /* Itti and Koch features */
    if(global_params.string["featureType"].compare("Itti")== 0)
    {
        /* Check that param keys exist */
        if(!init_params.num.count("nb_superpixels") || !init_params.num["downsampling_ratio"])
        {
            feature_extractor = new IttiFeatureExtractor();
        }
        else
        {
            feature_extractor = new IttiFeatureExtractor(init_params.num["downsampling_ratio"],
                                                         init_params.num["nb_superpixels"]);
        }
    }
    else if(global_params.string["featureType"].compare("Deep")== 0)
    {
        /* Initialize network */
        google::InitGoogleLogging("RL_IAC_SaliencyLearning");
#ifdef CPU_ONLY
        bool use_GPU = false;
#else
        bool use_GPU = true;
#endif
        if (use_GPU) {
            LOG(ERROR)<< "Using GPU";
            uint device_id = 0;
            LOG(ERROR) << "Using Device_id=" << device_id;
            Caffe::SetDevice(device_id);
            Caffe::set_mode(Caffe::GPU);
        }
        else {
            LOG(ERROR) << "Using CPU";
            Caffe::set_mode(Caffe::CPU);
        }
        /* Load deep feature parameters */
        string featureParamsFile = global_params.string["featureParamFile"];
        SaliencyFileParser::Param_Struct init_params;
        init_params.env_variables = global_params.env_variables;
        SaliencyFileParser fp;
        fp.parse_param_file(featureParamsFile, init_params);
        feature_extractor = new DeepFeatureExtractor(init_params.string["net_filename"],
                                                     init_params.string["weights_filename"],
                                                     init_params.string["mean_filename"],
                                                     init_params.string["extract_layer"],
                                                     init_params.num["nscales"],
                                                     init_params.num["nb_superpixels"]);

    }
    else
    {
        if(!init_params.num.count("nb_superpixels") || !init_params.num["downsampling_ratio"])
        {
            feature_extractor = new Make3DFeatureExtractor();
        }
        /* Load make3d parameters */

        feature_extractor = new Make3DFeatureExtractor(init_params.num["downsampling_ratio"],
                                                       init_params.num["nb_superpixels"],
                                                       init_params.num["make3d_scale"]);
    }
    return true;
}

bool SaliencyLearningExperiment::init_segmenter()
{
    SaliencyFileParser::Param_Struct seg_params;
    seg_params.env_variables = global_params.env_variables;
    SaliencyFileParser fp;
    if(! fp.parse_param_file(global_params.string["segParamPath"],seg_params))
        return false;


    object_segmenter = PtCldSegmentation<PointT>(seg_params.string["floor_equation_filename"],
                                                 seg_params.num["ransac_floor_dist_thresh"],
                                                 seg_params.num["normal_estim_max_depth"],
                                                 seg_params.num["normal_estim_smooth_size"],
                                                 seg_params.num["max_depth_visibility"],
                                                 seg_params.num["max_obj_dist_to_floor"],
                                                 seg_params.num["min_obj_dist_to_floor"],
                                                 seg_params.num["floor_angular_tolerance"],
                                                 seg_params.num["ransac_wall_dist_thresh"],
                                                 seg_params.num["min_wall_diag_size"],
                                                 seg_params.num["wall_angular_tolerance"],
                                                 seg_params.num["min_dist_btw_obj"],
                                                 seg_params.num["obj_min_diag_size"],
                                                 seg_params.num["obj_max_diag_size"],
                                                 seg_params.num["obj_at_border_pix_tolerance"],
                                                 seg_params.num["voxel_leaf_size"],
                                                 seg_params.num["max_obj_bottom_dist_to_floor"],
                                                 seg_params.num["min_obj_pixel_size"],
                                                 seg_params.num["use_tracking_option"],
                                                 seg_params.num["merge_clusters_option"],
                                                 seg_params.num["wrong_floor_thresh"]);
    return true;
}

bool SaliencyLearningExperiment::init_learner()
{
    /* parse */
    SaliencyFileParser::Param_Struct learn_params;
    learn_params.env_variables = global_params.env_variables;
    SaliencyFileParser fp;
    if(! fp.parse_param_file(global_params.string["learningParamPath"],
                             learn_params))
        return false;

    /* get a region learner list */
    std::vector<signed int> rll =  environment.get_region_learner_list();

    /* init object */
    vector<int> regions_weights;
    for(int i = 0 ; i < environment.get_nb_regions() ; i++)
    {
        regions_weights.push_back(1+environment.get_positions_per_region()[i].size());
    }
//    for(int i = 0 ; i < environment.get_nb_regions() ; i++)
//    {
//        regions_weights.push_back(1+environment.get_positions_per_region().size());
//    }
    cout << Mat(regions_weights).t() << endl;
    cout << Mat(rll).t() << endl;

    regions_learner.init_regions(  rll,regions_weights,
                                   learn_params.num["data_resampling_factor"],
                                   learn_params.num["data_ignore_unknowns"],
                                   learn_params.num["max_samples"],
                                   learn_params.num["samples_balanced_data"],
                                   learn_params.num["samples_random_replace"],
                                   learn_params.string["classifier_type"],
                                   learn_params.string["classifier_params"],
                                   global_params.num["evaluationMetrics"],
                                   global_params.num["usePerFrameEval"],
                                   global_params.num["useBackward"],
                                   global_params.num["intr_motivation_type"],
                                   global_params.num["alpha"]);
    if(global_params.num["useLongTerm"] == 1)
    {
        int nb_eval_frames = environment.get_inputs_size();
        int subsampling_rate = global_params.num["evalSubsamplingRate"];
        if(subsampling_rate == 0 )
            return false;

        regions_learner.init_long_term_eval(global_params.num["usePerRegionEval"]);
        for(int j = 0 ; j < nb_eval_frames ; j = j + subsampling_rate)
        {
            current_position = j;
            get_input();
            segment_objects();
            extract_features();
            int current_region = environment.get_region(current_position);
            regions_learner.add_long_term_image_data(current_region,
                                                     feature_map,
                                                     segmentation_mask);
        }
    }
    return true;
}

bool SaliencyLearningExperiment::init_action_selection()
{
    // if function is not called, selection will follow chronological order
    ExplorationParams params = ExplorationParams();
    params.RLIAC_RANDOM_POS_EPS = global_params.num["RNDPositionSelection"];
    params.RLIAC_RAND_SELECT_EPS = global_params.num["RNDActionSelection"];
    params.RLIAC_DISCOUNT_FACTOR = global_params.num["gamma"];

    action_selector.init(global_params.num["actionType"],
                         environment.get_world_map_graph(),
                         environment.get_positions_per_region(),
                         global_params.num["learnAtEachStep"],
                         global_params.num["nbLearnPerStay"],
                         global_params.num["learnMoveRatio"],
                         params);
}

bool SaliencyLearningExperiment::init_evaluation()
{
    /* Init evaluation enviornment */
    SaliencyFileParser fp;
    std::vector<std::string> inputs_list;
    std::map<std::string, int> input_region_map;
    WorldMapGraph::GraphStruct graph_struct;
    if(global_params.string["evalInputPath"].empty())
    {
        global_params.string["evalInputPath"] = global_params.string["inputPath"];
    }
    if(global_params.string["evalRegionPath"].empty())
    {
        global_params.string["evalRegionPath"] = global_params.string["inputRegionPath"];
    }

    /* try to load data from depth map names */
    fp.get_dir_file_names(global_params.string["evalInputPath"], inputs_list, "_depth", true);
    /* if failed, try to load data from GT map names */
    if(inputs_list.empty())
    {
        fp.get_dir_file_names(global_params.string["evalInputPath"], inputs_list, "_GT", true);
    }
    if(inputs_list.empty())
    {
        cerr << "No input data found in " << global_params.string["inputPath"] << endl;
    }

    if(!global_params.string["evalRegionPath"].empty())
    {
        fp.parse_input_region_file(global_params.string["evalRegionPath"], input_region_map);
    }
    if(!global_params.string["regionMapPath"].empty())
    {
        fp.parse_region_map_file(global_params.string["regionMapPath"], graph_struct);
    }
    experiment_eval.init_environment(global_params.string["evalInputPath"],
                                     inputs_list,input_region_map,graph_struct);

    /* Init evaluation set */
    cout << "Loading evaluation set ..." << endl;
    int subsampling_rate = global_params.num["evalSubsamplingRate"];
    if(subsampling_rate == 0 )
        return false;

    SaliencyFileParser::Param_Struct learn_params;
    learn_params.env_variables = global_params.env_variables;
    if(! fp.parse_param_file(global_params.string["learningParamPath"],learn_params))
        return false;

     experiment_eval.init_eval_set(0,learn_params.num["data_resampling_factor"],
                                   learn_params.num["data_ignore_unknowns"],
                                   subsampling_rate,
                                   global_params.num["evaluationMetrics"],
                                   global_params.num["usePerRegionEval"]);

     if(!bottom_up_extractor)
        experiment_eval.create_eval_set(feature_extractor);
     else
        experiment_eval.create_eval_set(bottom_up_extractor);

    /* Init log file */
     experiment_eval.init_log_file(global_params.string["outputLogFile"], params_filename);
     cout << "Done." << endl;
     return true;
}

bool SaliencyLearningExperiment::init_bottom_up()
{
    /* parse */
    SaliencyFileParser::Param_Struct bup_params;
    bup_params.env_variables = global_params.env_variables;
    SaliencyFileParser fp;
    if(! fp.parse_param_file(global_params.string["bottomupParamPath"],bup_params))
        return false;
    /* Get params */
    std::string param_file = bup_params.string["param_file"];
    std::string method_name = bup_params.string["method"];

    /* Get path for params (should be in "bottomupParamPath") */
    std::size_t path_end = global_params.string["bottomupParamPath"].find_last_of("/");
    std::string path = global_params.string["bottomupParamPath"];
    path.replace(path_end,path.size()-path_end,"/");
    param_file = path + param_file;

    /* Initialize saliency learner */
    if( method_name.compare("BMS") == 0)
        bottom_up_extractor = new saliencyMapBMS();
    else if( method_name.compare("Hou") == 0)
        bottom_up_extractor = new saliencyMapHou();
    else if( method_name.compare("Itti") == 0)
        bottom_up_extractor = new saliencyMapItti();
    else if( method_name.compare("Rudinac") == 0)
        bottom_up_extractor = new saliencyMapRudinac();
    else if( method_name.compare("VOCUS") == 0)
    {
        bottom_up_extractor = new VOCUS2();
        bottom_up_extractor->loadParam(param_file.c_str());
    }
    else
    {
        bottom_up_extractor = new VOCUS2();
        bottom_up_extractor->loadParam(param_file.c_str());
    }
    return true;
}

bool SaliencyLearningExperiment::init_missing_params()
{
    test_num_key("nbSteps", 150);
    test_num_key("initialPosition", 0);
    test_num_key("evaluationMetrics", 0);
    test_num_key("useSegImages", 1);
    test_num_key("useFloorTracker", 0);
    test_num_key("evaluationMetrics", 0);
    test_num_key("usePerFrameEval", 1);
    test_num_key("useLongTerm", 0);
    test_num_key("displayRegionState", 1);
    test_num_key("displayWorldMap", 1);
    test_num_key("usePerRegionEval", 1);
    test_num_key("evalSubsamplingRate", 10);
    test_num_key("actionType", ActionSelectionM::SELECT_CHRONO);
    test_num_key("useBackward",0);
    test_num_key("intr_motivation_type",0);
    test_num_key("nbLearnPerStay",2);
    test_num_key("learnAtEachStep",1);
    test_num_key("RNDActionSelection",0.1);
    test_num_key("RNDPositionSelection",0.1);
    test_num_key("learnMoveRatio",1);
    test_num_key("useSuperpixels",0);
    test_num_key("alpha",0.5);
    test_num_key("gamma",0.3);
    return true;
}

void SaliencyLearningExperiment::test_num_key(string key, float default_value)
{
    if(!global_params.num.count(key))
    {
        cout << "Key: " << key << "was not provided.";
        cout << "Using default " << default_value << endl;
        global_params.num[key] = default_value;
    }
}
