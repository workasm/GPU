#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

#define CV_IGNORE_DEBUG_BUILD_GUARD
#include <opencv2/opencv.hpp>

struct DisplayImageCV
{
    void show(const cv::Mat& input);

    static void mouseCallback(int /*event*/, int x, int y, int flags, void *data);
    static void JetColorImage(const cv::Mat &input, cv::Mat &out, double minFactor = 1.0,
                       double maxFactor = 1.0, bool flipRGB2BGR = true, bool useFloatImage = false);

    void processDebugBitmap(cv::Mat& out);

    std::vector<cv::Mat> m_chnImages;
};
