#include <algorithm>
#include <iostream>
#include <string>

#include <ros/ros.h>
#include <rosbag/view.h>

#include <gtest/gtest.h>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <dvs_msgs/EventArray.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#define foreach BOOST_FOREACH


char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main (int argc, char** argv)
{
    char* bagname = getCmdOption(argv, argv + argc, "-b");
    char* image_folder = getCmdOption(argv, argv + argc, "-i");
    char* skip_string = getCmdOption(argv, argv + argc, "-s");
    int skip_interval = std::stoi(skip_string);

    boost::filesystem::path p(bagname);

    ros::init (argc, argv, "bag_it");
    rosbag::Bag bag;
    bag.open(bagname, rosbag::bagmode::Read);
    mkdir(image_folder, 0777);

    int decay_const = 400;
    int total_i = 0;
    int idx = 0;
    double last_timestamp = 0;
    double temp_timestamp = 0;
    cv::Mat img(cv::Mat(120, 160, CV_32FC3, cv::Scalar(0,0,0)));
    for(rosbag::MessageInstance const m: rosbag::View(bag))
    {
        std::string eventType("/cam0/events");
        if (eventType.compare(m.getTopic()) != 0) continue;
        if ((++total_i) % skip_interval == 0) {
            // img.release();
            img = cv::Scalar::all(0.0f);
            continue;
        }

        // std::cout << p.stem() << ": "<< total_i << "         " << '\r';
        dvs_msgs::EventArray::ConstPtr msg = m.instantiate<dvs_msgs::EventArray>();
        // std::cout << brightness << std::endl;
        double brightness = (double) std::rand() / RAND_MAX / 5 + 0.8;

        if (msg != nullptr)
            for (int j=0; j<msg->events.size(); ++j){

                dvs_msgs::Event e = msg->events[j];
                temp_timestamp = e.ts.toSec();
                auto time_diff = temp_timestamp - last_timestamp;
                auto value = std::exp(-decay_const * time_diff);
                // double brightness = (double) std::rand() / RAND_MAX;

                if (e.x < 160 && e.y < 120){
                    Vec3f & color = img.at<Vec3f>(e.y,e.x);
                    if (e.polarity){
                        color[2] += value * brightness;
                    } else {
                        color[0] += value * brightness;
                    }
                }
            }
        
        boost::format fmt("%05d");

        // boost::filesystem::path dir(image_folder);
        // boost::filesystem::path filename((fmt % idx).str() + "_0.png");
        // boost::filesystem::path fullpath = dir / filename;
        // cv::imwrite(fullpath.string(), img * 255);


        // std::vector<cv::Mat> planes(3);
        // cv::split(img, planes); 
        // cv::normalize(planes[0], planes[0], 255, 0, cv::NORM_L2);
        // cv::normalize(planes[2], planes[2], 255, 0, cv::NORM_L2);
        // merge(planes,img);
        // std::cout << img << std::endl;
        // img *= 255;

        boost::filesystem::path dir(image_folder);
        boost::filesystem::path filename1((fmt % idx).str() + ".png");
        boost::filesystem::path fullpath1 = dir / filename1;
        cv::imwrite(fullpath1.string(), img * 255);


        auto time_diff = temp_timestamp - last_timestamp;
        auto value = std::exp(-decay_const * time_diff);
        // std::cout << value << std::endl;
        // std::cout << " " << std::endl;
        // std::cout << img << std::endl;
        img *= value;
        // std::cout << " " << std::endl;
        // std::cout << img << std::endl;
        // std::cout << " " << std::endl;
        // std::cout << " " << std::endl;
        // break;
        last_timestamp = temp_timestamp;
        idx++;



    }

    std::cout << p.stem() << ": "<< idx << "         " << std::endl;
    bag.close();
    return (0);
}