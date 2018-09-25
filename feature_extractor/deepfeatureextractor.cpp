/**
 * \file deepfeatureextractor.cpp
 * \brief DeepFeatureExtractor
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 14 / 2016
 *
 * Custom DOxygen-style template. Provide here a description of the file
 *
 */

#include "deepfeatureextractor.h"
//#include "../Utils/Optimize.h"
#include "highgui.h"
#include "printdebug.h"

#define BATCH_SIZE 1
using namespace cv;

DeepFeatureExtractor::DeepFeatureExtractor():FeatureExtractor()
{
    INIT = false;
}

DeepFeatureExtractor::DeepFeatureExtractor(int downsampling_ratio, int nb_superpixels)
                     :FeatureExtractor(downsampling_ratio, nb_superpixels)
{
    INIT = false;
}


/**
 * @brief calculate Deep features maps V2
 * @param input
 * @return the vector of feature maps
 */
std::vector<Mat> DeepFeatureExtractor::getFeatureMap(Mat input)
{
    std::vector<Mat> feature_maps;
    if(INIT)
    {
        if(input.empty())
        {
            return feature_maps;
        }
        //resize(input,input, Size(input.cols*2,input.rows*2));

        Blob<float>* input_image_layer = net->input_blobs()[0];
        int num_channels = input_image_layer->channels();
        input_image_layer->Reshape(BATCH_SIZE, num_channels, (int)(input.rows), (int)(input.cols));

        net->Reshape();
        _mean = SetMean(net);
        cv::Mat features = ExtractDeepFeatures(input,extract_layer);
        cv::split(features, feature_maps);
    }
    return feature_maps;
}

int DeepFeatureExtractor::getNbFeatures()
{
    return 256;
}

int DeepFeatureExtractor::getFeatureType()
{
    return DEEP;
}



/////////////////////////////////////////////////
///
///



DeepFeatureExtractor::DeepFeatureExtractor(const std::string& model_file,
                                           const std::string& trained_file,
                                           const std::string& mean_file,
                                           const std::string& extract_layer,
                                           const int nscales,
                                           const int nb_superpixels)
:FeatureExtractor(1, nb_superpixels)
{

    /* Load the network. */
    printf("Loading VPS network...\n");
    std::cout << "network " << model_file << std::endl;
    net.reset(new Net<float>(model_file,caffe::TEST)); //TEST
    net->CopyTrainedLayersFrom(trained_file);

//    _mean = SetMean(net);
    SetNetworkParams(net);

    if(mean_file.empty())
        use_mean = false;
    else
    {
        std::ifstream inp(mean_file.c_str());
        inp >> meanRGB[0] >> meanRGB[1] >> meanRGB[2];
        use_mean = true;
    }
    this->extract_layer = extract_layer;
    this->nscales = nscales;
    INIT = true;
}

void DeepFeatureExtractor::SetNetworkParams(caffe::shared_ptr<Net<float> > net) {
//	CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_image_layer = net->input_blobs()[0];
    int num_channels = input_image_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1)
            << "Input layer should have 1 or 3 channels.";

    cv::Size input_geometry = cv::Size(input_image_layer->width(), input_image_layer->height());

//    input_image_layer->Reshape(BATCH_SIZE, num_channels, input_geometry.height, input_geometry.width);
    if (net->input_blobs().size() > 1)
    {
        Blob<float>* label_layer = net->input_blobs()[1];
        label_layer->Reshape(BATCH_SIZE, 1, 1, 1);
    }

    net->Reshape();
    if(use_mean)
        _mean = SetMean(net);
    INIT = true;

    //caffe::fb::optimizeMemory(net.get());
}

/* Load the mean file in binaryproto format. */
cv::Mat DeepFeatureExtractor::SetMean(caffe::shared_ptr<Net<float> > net) {
    cv::Mat mean, mean_float;

    Blob<float>* input_layer = net->input_blobs()[0];
    cv::Size input_geometry = cv::Size(input_layer->width(), input_layer->height());

    std::vector<cv::Mat> channels(3);
    channels[0] = cv::Mat(input_geometry, mean.type(), meanRGB[0]);
    channels[1] = cv::Mat(input_geometry, mean.type(), meanRGB[1]);
    channels[2] = cv::Mat(input_geometry, mean.type(), meanRGB[2]);

    cv::merge(channels, mean);

    mean.convertTo(mean_float, CV_32FC1);

    return mean_float;
}

cv::Mat DeepFeatureExtractor::RunNetwork(caffe::shared_ptr<Net<float> > net, cv::Mat &meanImg, cv::Mat &img, std::string layer_name)
{
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(net, &input_channels);

    Preprocess(img, meanImg, &input_channels);

    net->ForwardPrefilled();

    const caffe::shared_ptr<Blob<float> > output_layer = net->blob_by_name(layer_name);
    const float* begin = output_layer->cpu_data();
    int nchannels = output_layer->channels();
    if(output_layer->channels() > CV_CN_MAX)
        nchannels = output_layer->channels()/2;

    cv::Mat output_mat(nchannels, output_layer->height()*output_layer->width(), CV_32F);
    output_mat.data = (uchar*)begin;
    output_mat = output_mat.t();
    output_mat = output_mat.reshape(nchannels,output_layer->height());

    return output_mat;
}


cv::Mat DeepFeatureExtractor::ExtractDeepFeatures(cv::Mat &img, std::string layer_name)
{
    return RunNetwork(net,_mean, img, layer_name);
}

std::vector<Mat> DeepFeatureExtractor::pad_images(cv::Mat input, int padding)
{
    cout << "Padding" << endl;
    std::vector<Mat> padded_input;
    /* pad images */
    cv::Mat input1, input2, input3, input4;
    /// top left
    input1 = input.clone();
    padded_input.push_back(input1);

    /// top right
    input(cv::Rect(0,0,input.cols-padding, input.rows)).copyTo(input2);
    cv::copyMakeBorder( input2, input2, 0, 0, padding, 0,cv::BORDER_REFLECT );
    padded_input.push_back(input2);

    /// bottom left
    input(cv::Rect(0,0,input.cols, input.rows-padding)).copyTo(input3);
    cv::copyMakeBorder( input3, input3,padding, 0, 0, 0,cv::BORDER_REFLECT );
    padded_input.push_back(input3);

    /// bottom right
    input(cv::Rect(0,0,input.cols-padding, input.rows-padding)).copyTo(input4);
    cv::copyMakeBorder( input4, input4,padding, 0, padding, 0,cv::BORDER_REFLECT );
    padded_input.push_back(input4);
    cout << "Padding finished" << endl;
    return padded_input;

}

Mat DeepFeatureExtractor::reformat_feature_maps(std::vector<Mat> padded_features)
{
    cout << "formatting" << endl;
    std::vector<cv::Mat> inputs;
    int c = padded_features[0].cols;
    int r = padded_features[0].rows;
    int ch = padded_features[0].channels();
    cv::Mat input1,input2,input3,input4;
    input1 = padded_features[0].reshape(ch,c*r);
    input2 = padded_features[1].reshape(ch,c*r);
    input3 = padded_features[2].reshape(ch,c*r);
    input4 = padded_features[3].reshape(ch,c*r);
    inputs.push_back(input1);
    inputs.push_back(input2);
    cv::Mat output;
    cv::hconcat(inputs,output);

    std::vector<cv::Mat> inputs2;
    inputs2.push_back(input3);
    inputs2.push_back(input4);
    cv::Mat output2;
    cv::hconcat(inputs2,output2);

    output = output.reshape(ch, r);
    output2 = output2.reshape(ch,r);
    vector<Mat> v_output1, v_output2, v_output3;
    split(output, v_output1);
    split(output2, v_output2);
    for(int i = 0 ;  i < v_output1.size() ; i++)
    {
        v_output1[i] = v_output1[i].t();
        v_output2[i] = v_output2[i].t();
    }
    merge(v_output1, output);
    merge(v_output2, output2);
    output = output.reshape(ch, 2*r*c);
    output2 = output2.reshape(ch, 2*r*c);
    std::vector<cv::Mat> inputs3;
    inputs3.push_back(output);
    inputs3.push_back(output2);
    cv::Mat output3;
    cv::hconcat(inputs3,output3);
    output3 = output3.reshape(ch, c*2);
    split(output3, v_output3);
    for(int i = 0 ;  i < v_output3.size() ; i++)
    {
        v_output3[i] = v_output3[i].t();
    }
    merge(v_output3, output3);
    output3 = output3.reshape(ch, r*2);

    return output3;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void DeepFeatureExtractor::WrapInputLayer(caffe::shared_ptr<Net<float> > net, std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void DeepFeatureExtractor::Preprocess(const cv::Mat& img, cv::Mat& meanImg, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    int num_channels_ = input_channels->size();
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Size input_geometry = cv::Size(input_channels->at(0).cols, input_channels->at(0).rows);

    cv::Mat sample_resized;
    if (sample.size() != input_geometry)
        cv::resize(sample, sample_resized, input_geometry);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if(use_mean)
        cv::subtract(sample_float, meanImg, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
}
