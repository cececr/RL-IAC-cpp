/**
 * \file main.cpp
 * \brief
 * \author Céline Craye
 * \version 0.1
 * \date 12 / 8 / 2015
 *
 * Saliency learning using RL-IAC
 *
 */
#include "saliencylearningexperiment.h"
#include "iostream"

using namespace std;

int main(int argc, const char* argv[])
{

    std::srand(std::time(NULL));
    /* create experiment */
    SaliencyLearningExperiment* exp;
    std::string param_file;
    if(argc > 1)
    {
        param_file = std::string(argv[1]);
    }
    else
    {
        param_file = std::string("../RL_IAC_SaliencyLearning/params.dat");
    }
    std::cout << param_file << std::endl;
    exp =  new SaliencyLearningExperiment(param_file);

    /* run ! */
    if(argc <= 1)
    {
//        exp->run_segmentation_only();
        exp->run();
//        exp->run_bottom_up();
//        exp->run_offline_learning();
//        exp->run_offline_saliency();
    }
    else
    {
        std::string run_type = "RL_IAC";
        if(argc > 2)
        {
            run_type = std::string(argv[2]);
        }


        if(run_type.compare("RL_IAC") == 0)
        {
            std::cout << "RL_IAC run" << std::endl;
            exp->run();
        }
        else if(run_type.compare("offline_saliency") == 0)
        {
            std::cout << "Offline saliency run" << std::endl;
            exp->run_offline_saliency();
        }
        else if(run_type.compare("offline_learning") == 0)
        {
            std::cout << "Offline learning run" << std::endl;
            exp->run_offline_learning();
        }
        else if(run_type.compare("bottom_up") == 0)
        {
            std::cout << "Bottom-up run" << std::endl;
            exp->run_bottom_up();
        }
        else if(run_type.compare("segmentation_only") == 0)
        {
            std::cout << "segmentation run" << std::endl;
            exp->run_segmentation_only();
        }
        else
        {
            std::cerr << "Invalid parameters." << std::endl;
            std::cout << "Call should be done this way" << std::endl;
            std::cout << "./RL_IAC_SaliencyLearning <param_file> <run_type>" << std::endl;
            std::cout << "<run_type> = " << std::endl;
            std::cout << "             - RL_IAC : basic usage" << std::endl;
            std::cout << "             - offline_saliency : evaluate saliency with offline model" << std::endl;
            std::cout << "             - offline_learning : learn an offline model of saliency" << std::endl;
            std::cout << "             - bottom_up : evaluate bottom-up saliency" << std::endl;
            std::cout << "             - segmentation_only : segment and create output masks" << std::endl;
        }

    }
    delete exp;
    return 0;
}


