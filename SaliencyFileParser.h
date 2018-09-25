/**
 * \file SaliencyFileParser.h
 * \brief SaliencyFileParser
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 21 / 2015
 *
 * File parser for algorithm parameters.
 * Also contains a directory file lister for linux system.
 * TODO: - Add load model option to parser
 */


#ifndef SALIENCYFILEPARSER_H
#define SALIENCYFILEPARSER_H
#include <iostream>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atof */
#include <fstream>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <vector>
#include <algorithm> /* sort */
#include <map>
class SaliencyFileParser
{
public:

    struct Param_Struct
    {
        std::map<std::string, std::string> string;
        std::map<std::string, float> num;
        std::map<std::string, std::string> env_variables;
    };

    SaliencyFileParser();
    bool parse_param_file(std::string param_file,
                          Param_Struct &params);
    bool get_dir_file_names(std::string dir_name,
                            std::vector<std::string> &file_names,
                            std::string root_name = "", bool remove_root = false);
    int parse_input_region_file(std::string input_region_file, std::map<std::string, int>& input_region_map);
    bool parse_region_map_file(std::string region_map_file,
                               std::vector<std::vector<std::vector<int> > > &graph_struct);
    bool saveInFile(std::ofstream & logfile, std::string param_file);
    std::string get_env_var(std::string key);


private:
    static const int ROS_PARSE_TYPE = 0;
    static const int PARAM_PARSE_TYPE = 1;
    static const int VOID_TYPE = 0;
    static const int STRING_TYPE = 1;
    static const int NUM_TYPE = 2;
    int parse_line(std::string line, int parse_type, std::string& key,std::string& value);
    void add_root(std::string& str);

    std::map<std::string, std::string> env_variables;


};

#endif // SALIENCYFILEPARSER_H
