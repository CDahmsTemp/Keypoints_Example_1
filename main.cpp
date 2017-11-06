// main.cpp

#include<opencv2/opencv.hpp>

#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows

// global variables ///////////////////////////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BRIGHT_GREEN = cv::Scalar(0.0, 255.0, 0.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<cv::DMatch> findGoodKeypointMatches(const std::vector<cv::DMatch> &allMatches, const cv::Mat &matModelDescriptors);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    // open the model image as grayscale, if unsuccessful show an error message and bail
    cv::Mat imgModel = cv::imread("model_1.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (imgModel.empty()) {
        std::cout << "error: model image not read from file\n\n";
        _getch();                           // may have to modify this line if not using Windows
        return(0);
    }

    // open the scene image as grayscale, if unsuccessful show an error message and bail
    cv::Mat imgScene = cv::imread("scene_1.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (imgScene.empty()) {
        std::cout << "error: scene image not read from file\n\n";
        _getch();                           // may have to modify this line if not using Windows
        return(0);
    }

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> modelKeypoints;
    std::vector<cv::KeyPoint> sceneKeypoints;
    cv::Mat matModelDescriptors;
    cv::Mat matSceneDescriptors;

    // call detectAndCompute() for the model and the scene    
    akaze->detectAndCompute(imgModel, cv::noArray(), modelKeypoints, matModelDescriptors);
    akaze->detectAndCompute(imgScene, cv::noArray(), sceneKeypoints, matSceneDescriptors);

    // declare a matcher and perform the match
    cv::BFMatcher bfMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    bfMatcher.match(matModelDescriptors, matSceneDescriptors, matches);

    // find good keypoint matches, this line can be commented out if all matches are desired
    matches = findGoodKeypointMatches(matches, matModelDescriptors);

    // declare an image to draw the model, scene, and matches on, then draw the matches
    // note that imgMatches is a color image even though imgModel and imgScene were read in as grayscale above, this is so we can draw stuff on imgMatches in color
    cv::Mat imgMatches;
    cv::drawMatches(imgModel, modelKeypoints, imgScene, sceneKeypoints, matches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // note: you could end the program here if the homography box isn't needed

    // get the matching model and scene keypoints into vectors
    std::vector<cv::Point2f> modelMatchingKeypoints;
    std::vector<cv::Point2f> sceneMatchingKeypoints;
    for (int i = 0; i < matches.size(); i++) {
        //-- Get the keypoints from the good matches
        modelMatchingKeypoints.push_back(modelKeypoints[matches[i].queryIdx].pt);
        sceneMatchingKeypoints.push_back(sceneKeypoints[matches[i].trainIdx].pt);
    }

    // call findHomography()
    cv::Mat matHomography = cv::findHomography(modelMatchingKeypoints, sceneMatchingKeypoints, CV_RANSAC);

    // declare and populate a vector of the model's corners
    std::vector<cv::Point2f> modelCorners(4);
    modelCorners[0] = cv::Point(0, 0);
    modelCorners[1] = cv::Point(imgModel.cols, 0);
    modelCorners[2] = cv::Point(imgModel.cols, imgModel.rows);
    modelCorners[3] = cv::Point(0, imgModel.rows);

    // declare and find the corners of the detected object in the scene
    std::vector<cv::Point2f> foundObjectCorners(4);
    cv::perspectiveTransform(modelCorners, foundObjectCorners, matHomography);

    // calculate the corners of the found object in the scene
    // note that since the model is on the left side of the matches image we have to add an offset by that amount
    cv::Point ptTopLeft(foundObjectCorners[0] + cv::Point2f(imgModel.cols, 0));
    cv::Point ptTopRight(foundObjectCorners[1] + cv::Point2f(imgModel.cols, 0));
    cv::Point ptBottomRight(foundObjectCorners[2] + cv::Point2f(imgModel.cols, 0));
    cv::Point ptBottomLeft(foundObjectCorners[3] + cv::Point2f(imgModel.cols, 0));

    // draw lines between the corners of the found object in the scene
    cv::line(imgMatches, ptTopLeft, ptTopRight, SCALAR_BRIGHT_GREEN, 3);
    cv::line(imgMatches, ptTopRight, ptBottomRight, SCALAR_BRIGHT_GREEN, 3);
    cv::line(imgMatches, ptBottomRight, ptBottomLeft, SCALAR_BRIGHT_GREEN, 3);
    cv::line(imgMatches, ptBottomLeft, ptTopLeft, SCALAR_BRIGHT_GREEN, 3);

    //-- Show detected matches
    cv::imshow("imgMatches", imgMatches);

    cv::waitKey(0);
    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<cv::DMatch> findGoodKeypointMatches(const std::vector<cv::DMatch> &allMatches, const cv::Mat &matModelDescriptors) {
    std::vector<cv::DMatch> goodMatches;        // this will be the return value

                                                // find the min and max distances between keypoints
    double minDistance = 10000.0;
    double maxDistance = 0.0;
    for (int i = 0; i < matModelDescriptors.rows; i++) {
        double distance = allMatches[i].distance;
        if (distance < minDistance) minDistance = distance;
        if (distance > maxDistance) maxDistance = distance;
    }
    std::cout << "minDistance = " << minDistance << "\n";
    std::cout << "maxDistance = " << maxDistance << "\n";

    // find good matches, i.e. distance is less than threshold based on min distance
    // ToDo: the 3.0 threshold should be a function parameter
    for (int i = 0; i < matModelDescriptors.rows; i++) {
        if (allMatches[i].distance < (3.0 * minDistance)) {
            goodMatches.push_back(allMatches[i]);
        }
    }
    return(goodMatches);
}

