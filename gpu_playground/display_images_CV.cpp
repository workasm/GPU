
#include <iostream>

#include "display_images_CV.h"

void DisplayImageCV::show(const cv::Mat& input) {

    cv::destroyAllWindows();
    std::string wndName("main");

    cv::namedWindow(wndName, cv::WINDOW_KEEPRATIO);
    cv::split(input, m_chnImages);

    cv::Mat out;
    for(size_t i = 0; i < m_chnImages.size(); i++) {

        JetColorImage(m_chnImages[i], out);
        if(i < m_chnImages.size() - 1) {
            auto name = std::string("chn#") + std::to_string(i);
            cv::namedWindow(name, cv::WINDOW_KEEPRATIO);
            cv::imshow(name, out);
        } else {
            cv::imshow(wndName, out);
            //cv::setMouseCallback(wndName, mouseCallback, this);
        }
    }
    cv::waitKey();
}

void DisplayImageCV::mouseCallback(int /*event*/, int x, int y, int, void *data)
{
    auto self = (DisplayImageCV *)data;

    const auto &image = self->m_chnImages.back();
    int w = image.cols, h = image.rows;
    float dx = x - w * 0.5f, dy = y - h * 0.5f,
          R = std::sqrt(dx * dx + dy * dy);

    fprintf(stderr, "I[%d,%d] = %f; R = %f\n",
            x, y, image.ptr<float>(y)[x], R);
}

void DisplayImageCV::JetColorImage(const cv::Mat &input, cv::Mat &out, double minFactor, double maxFactor,
                   bool flipRGB2BGR, bool useFloatImage)
{
    if (input.channels() != 1)
        throw std::runtime_error("Only images with a single channel can be processed !");

    cv::Mat fpImage = input;
    if (!useFloatImage)
        input.convertTo(fpImage, CV_64FC1);

    out.create(input.size(), CV_8UC3);
    out = cv::Scalar(0);

    double minVal, maxVal;
    cv::minMaxIdx(fpImage, &minVal, &maxVal);

    minFactor = std::min(std::abs(minFactor), 1.0);
    maxFactor = std::min(std::abs(maxFactor), 1.0);
    minVal *= minFactor, maxVal *= maxFactor;

    const int r = flipRGB2BGR ? 2 : 0,
              b = flipRGB2BGR ? 0 : 2;

    auto range = (maxVal - minVal);
    if (range <= 1e-6)
    {
        //fprintf(stderr, "Invalid min/max range: [%f; %f]!\n", minVal, maxVal);
        memset(out.ptr(), 0xCC, out.step * out.rows);
        cv::line(out, {0, 0}, {out.cols, out.rows}, cv::Scalar(255, 0, 0), 5);
        cv::line(out, {0, out.rows}, {out.cols, 0}, cv::Scalar(255, 0, 0), 5);
        return;
    }

    for (int y = 0; y < fpImage.rows; y++)
    {
        auto p_in = (useFloatImage ? (const void *)fpImage.ptr<float>(y) : (const void *)fpImage.ptr<double>(y));
        auto p_out = out.ptr(y);
        for (int x = 0; x < fpImage.cols; x++, p_out += 3)
        {
            double pval = (useFloatImage ? ((const float *)p_in)[x] : ((const double *)p_in)[x]);
            if (!std::isfinite(pval))
                continue;

            auto gray = std::min(pval, maxVal);
            gray = 8.0 * (std::max(gray, minVal) - minVal) / range;
            const double s = 255.0 / 2.0;

//            gray = 7; // HACK HACK

            if (gray <= 1)
            {
                p_out[b] = (uint8_t)cvRound((gray + 1) * s);
            }
            else if (gray <= 3)
            {
                p_out[1] = (uint8_t)cvRound((gray - 1) * s);
                p_out[b] = 255;
            }
            else if (gray <= 5)
            {
                p_out[r] = (uint8_t)cvRound((gray - 3) * s);
                p_out[1] = 255;
                p_out[b] = (uint8_t)cvRound((5 - gray) * s);
            }
            else if (gray <= 7)
            {
                p_out[r] = 255;
                p_out[1] = (uint8_t)cvRound((7 - gray) * s);
            }
            else
            {
                p_out[r] = (uint8_t)cvRound((9 - gray) * s);
            }
        } // for x
    } // for y
}

