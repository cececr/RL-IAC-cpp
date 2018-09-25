/**
 * \file SaliencyFileParser.cpp
 * \brief SaliencyFileParser
 * \author Céline Craye
 * \version 0.1
 * \date 4 / 21 / 2015
 *
 * File parser for algorithm parameters. Also contains a directory lister for linux system.
 *
 */


#include "SaliencyFileParser.h"

#include <cctype>
#include <string>


using namespace std;

SaliencyFileParser::SaliencyFileParser()
{
}

bool SaliencyFileParser::parse_param_file(string param_file,
                                    Param_Struct &params)
{
    this->env_variables = params.env_variables;
    /* Read param file and extract settings */
    ifstream file(param_file.c_str(), ios::in);
    /* Get parse type */
    int parse_type;
    (param_file.find(".yaml")!=std::string::npos) ?
                parse_type = ROS_PARSE_TYPE : parse_type = PARAM_PARSE_TYPE;

    if(file)
    {
        string line;
        while(getline(file, line))
        {

            std::string key;
            std::string value;
            int value_type;
            value_type = parse_line(line, parse_type, key, value);
            if(value_type == VOID_TYPE)
                continue;
            if(value_type == STRING_TYPE)
                params.string[key] = value;
            if(value_type == NUM_TYPE)
            {
                float numvalue = atof(value.c_str());
                params.num[key] = numvalue;
            }
        }
        file.close();
        /* Add environment variables */
        typedef std::map<std::string, std::string>::iterator it_type;
        for(it_type iterator = params.string.begin(); iterator != params.string.end(); iterator++) {
            add_root(iterator->second);
        }

        params.env_variables = this->env_variables;

        return true;
    }
    else
    {
        cerr << "FileParser: Cannot open param file " << param_file << endl;
        return false;
    }

}

bool is_space(char x)
{
    return std::isspace(x);
}

int SaliencyFileParser::parse_line(string line, int parse_type, string &key, string &value)
{
    size_t found;
    if(parse_type == ROS_PARSE_TYPE)
    {
        /* if # found, this is a comment */
        found = line.find("#");
        if (found!=std::string::npos)
            return VOID_TYPE;
        /* find { and } and remove */
        found = line.find("{");
        if (found!=std::string::npos)
            line = line.substr(found + 1);
        found = line.find("}");
        if (found!=std::string::npos)
            line = line.substr(0,found-1);
        /* find : */
        found = line.find(":");
        if (found==std::string::npos)
            return VOID_TYPE;
        else
        {
            /* remove blanks and get key */
            key = line.substr(0,found);
            key.erase(remove_if(key.begin(),
                                key.end(),
                                is_space),key.end());
        }
        /* find */
        line = line.substr(found+1);
        found = line.find(",");
        value = line.substr(0,found);
        /* if " found, store value and return string type */
        found = value.find("\"");
        if (found!=std::string::npos)
        {
            value = value.substr(found+1);
            found = value.find("\"");
            if (found!=std::string::npos)
            {
                value = value.substr(0,found);
                return STRING_TYPE;
            }
            else
            {
                return VOID_TYPE;
            }
        }
        /* else, remove blanks, this is a float type */
        else
        {
            value.erase(remove_if(value.begin(),
                              value.end(),
                              is_space), value.end());
            return NUM_TYPE;
        }
    } // end ROS_PARSE_TYPE
    else
    {
        found = line.find("#");
        if (found!=std::string::npos)
        {
            return VOID_TYPE;
        }

        /* get environment variable */
        found = line.find("$");
        if (found<2)
        {
            key = line.substr(found);
            found = line.find("=");
            if (found==std::string::npos)
                return VOID_TYPE;
            key = key.substr(0,found-1);
            value = line.substr(found+1);
            found = value.find("\"");
            if (found!=std::string::npos)
            {
                value = value.substr(found+1);
                found = value.find("\"");
                if (found!=std::string::npos)
                {
                    value = value.substr(0,found);
                    env_variables[key] = value;
                }
            }

            return VOID_TYPE;
        }
        /* get key */
        found = line.find("[");
        if (found==std::string::npos)
            return VOID_TYPE;
        key = line.substr(found+1);
        found = key.find("]");
        if (found==std::string::npos)
            return VOID_TYPE;
        key = key.substr(0,found);
        key.erase(remove_if(key.begin(),
                          key.end(),
                          is_space), key.end());

        /* get value */
        found = line.find("=");
        if (found==std::string::npos)
            return VOID_TYPE;
        value = line.substr(found+1);
        /* if " found, store value and return string type */
        found = value.find("\"");
        if (found!=std::string::npos)
        {
            value = value.substr(found+1);
            found = value.find("\"");
            if (found!=std::string::npos)
            {
                value = value.substr(0,found);
                return STRING_TYPE;
            }
            else
            {
                return VOID_TYPE;
            }
        }
        /* else, remove blanks, this is a float type */
        else
        {
            value.erase(remove_if(value.begin(),
                              value.end(),
                              is_space), value.end());
            if(value.empty())
            {
                return VOID_TYPE;
            }
            return NUM_TYPE;
        }
    }
}

void SaliencyFileParser::add_root(string &str)
{
    typedef std::map<std::string, std::string>::iterator it_type;
    for(it_type iterator = env_variables.begin(); iterator != env_variables.end(); iterator++)
    {
        int found = str.find(iterator->first);
        if (found!=std::string::npos)
        {
            int found2 = str.find("/");
            if(found!=std::string::npos)
            {
                str.replace(str.begin()+found, str.begin()+found2,iterator->second);
            }
        }
    }
}

/**
 * @brief SaliencyFileParser::saveInFile write a header with params in given file
 * @param logfile
 * @return true if params were successfuly written
 */
bool SaliencyFileParser::saveInFile(ofstream &logfile, string param_file)
{
    if(!logfile)
    {
        cerr << "FileParser: Could not write into log file" << endl;
        return false;
    }
    /* Read param file and extract settings */
    ifstream file(param_file.c_str(), ios::in);
    if(file)
    {
        string line;
        size_t found;
        while(getline(file, line))
        {
            found = line.find("#");
            if (found!=std::string::npos)
            {
                // Then the line is a comment and must be ignored
                continue;
            }
            logfile << line << endl;
        }
        file.close();
        return true;
    }
    else
    {
        cerr << "FileParser: Cannot open param file " << param_file << endl;
        return false;
    }

    return true;
}

string SaliencyFileParser::get_env_var(string key)
{
    return env_variables[key];
}


bool SaliencyFileParser::get_dir_file_names(string dir_name,
                                            std::vector<string> &file_names,
                                            string root_name, bool remove_root)
{
    DIR *dir = NULL;
    struct dirent *file = NULL;

    if((dir = opendir(dir_name.c_str())) == NULL)
    {
        return EXIT_FAILURE;
    }

    while((file = readdir(dir)) != NULL)
    {
        if(strcmp(file->d_name, ".") && strcmp(file->d_name, ".."))
        {
            string filename(file->d_name);
            size_t strpos = filename.find(root_name);
            if(strpos!=string::npos)
            {
                if(remove_root == true)
                {
                    filename.replace(strpos,root_name.size(),"");
                }
                file_names.push_back(filename);
            }
        }
    }
    sort( file_names.begin(), file_names.end() );
    return file_names.size();
}

int SaliencyFileParser::parse_input_region_file(std::string input_region_file,
                                           std::map<string, int> &input_region_map)
{
    /* Open cluster filename */
    ifstream file(input_region_file.c_str(), ios::in);
    if(!file)
    {
        cerr << "FileParser: Cannot open cluster file " << input_region_file << endl;
        return 0;
    }

    /* Read param file and extract settings */
    string line;
    vector<int> clusters_vect;
    while(getline(file, line))
    {
        string key;
        string value;
        parse_line(line, PARAM_PARSE_TYPE, key, value);
        int num_value = atoi(value.c_str());
        input_region_map[key] = num_value;
        clusters_vect.push_back(num_value);
    }
    file.close();

    /* Update number of clusters */
    int nbClusters;
    if(clusters_vect.size() > 0)
    {
        sort( clusters_vect.begin(), clusters_vect.end() );
        vector<int>::iterator it;
        it = std::unique (clusters_vect.begin(), clusters_vect.end());
        clusters_vect.resize( std::distance(clusters_vect.begin(),it) );
        nbClusters = clusters_vect.size();
    }

    return nbClusters;
}

bool SaliencyFileParser::parse_region_map_file(string region_map_file, vector<vector<vector<int> > > & graph_struct)
{
    /* Open filename */
    ifstream file(region_map_file.c_str(), ios::in);
    if(!file)
    {
        cerr << "FileParser: Cannot open cluster file " << region_map_file << endl;
        return false;
    }

    string line;
    size_t found;
    while(getline(file, line))
    {
        found = line.find("#");
        if (found!=std::string::npos)
        {
            // Then the line is a comment and must be ignored
            continue;
        }
        found = line.find("=");
        if (found!=std::string::npos)
        {
            line = line.substr(found+1);
            vector<vector<int> > node;
            while(line.find("[")!=std::string::npos)
            {
                found = line.find("[");
                line = line.substr(found+1);
                found = line.find("]");
                string vect_string =line.substr(0,found);
                vector<int> vect_data;
                while(vect_string.find(",")!=std::string::npos)
                {
                    found = vect_string.find(",");
                    string val_string = vect_string.substr(0,found);
                    vect_data.push_back(atoi(val_string.c_str()));
                    vect_string = vect_string.substr(found+1);
                }
                vect_data.push_back(atoi(vect_string.c_str()));
                node.push_back(vect_data);
            }
            graph_struct.push_back(node);
        }
    }
    file.close();
    return true;
}

