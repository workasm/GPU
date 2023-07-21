
#ifndef INTERPOLATE_HOST_H
#define INTERPOLATE_HOST_H

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <cmath>

#include "macros.h"
#include "common_types.h"

// inverse distance weighting interpolation
template <class InputReal, class OutputReal>
class DataInterpolator
{
    // minimal angle range for outer radius points
    static constexpr OutputReal minAngleRange = (OutputReal)M_PI;

public:
    using Params = InterpParams< OutputReal >;
    using Pix = InterpPix< OutputReal >;

private:
    bool m_isReset = false;
    OutputReal *m_outPtr = nullptr; // output buffer pointer where final data is to be written
    Params m_params{};     // algorithm parameters
    TImage<Pix> m_pixels;

    std::vector<InputReal> m_scanPts;
    std::unique_ptr<std::ofstream> m_pofs;

    uint32_t m_outerPts = 0, m_innerPts = 0;

public:
    DataInterpolator() = default;

    CLASS_NO_COPY(DataInterpolator);

    DataInterpolator &operator=(DataInterpolator &&rhs) noexcept
    {
        if (this != &rhs)
        {
            this->swap(rhs);
        }
        return *this;
    }
    DataInterpolator(DataInterpolator &&rhs) noexcept
    {
        *this = std::move(rhs);
    }

    bool getReset() const
    {
        return m_isReset;
    }

    void setup(OutputReal *dataOut, const Params &params)
    {
        m_outPtr = dataOut, m_params = params;
        XPRINTZ("Setup: data %p; %d x %d; numSignals: %d; rad: [%f %f] - [%f %f] ",
                dataOut, m_params.w, m_params.h, m_params.numSignals,
                m_params.innerRadX, m_params.innerRadY, m_params.outerRadX, m_params.outerRadY);

        m_pixels.resize(params.w * m_params.numSignals, params.h);
        memset(m_pixels.data(), 0, m_pixels.sizeInBytes());
        m_scanPts.clear();
        m_scanPts.reserve(params.w * params.h * (m_params.numSignals + 2));
        m_isReset = true;
    }

    void swap(DataInterpolator &rhs) noexcept
    {
        std::swap(m_isReset, rhs.m_isReset);
        std::swap(m_outPtr, rhs.m_outPtr);
        std::swap(m_params, rhs.m_params);
        m_pixels.swap(rhs.m_pixels);
        m_scanPts.swap(rhs.m_scanPts);
        m_pofs.swap(rhs.m_pofs);
        std::swap(m_outerPts, rhs.m_outerPts);
        std::swap(m_innerPts, rhs.m_innerPts);
    }

    // z format: [x, y, sig1, sig2, ...] total of m_params.numSignals + 2
    void addPoint(const InputReal *z) noexcept
    {
        size_t totalS = m_params.numSignals + 2, zi = 2; // skip first x/y signals
        for (; zi < totalS; zi++)
        {
            if (std::isfinite(z[zi]))
                break;
        }
        if (zi >= totalS) // no data signal with a finite value found => drop it
            return;

        m_scanPts.insert(m_scanPts.end(), z, z + totalS);

        if (m_pofs)
        {
            m_pofs->write((const char *)z, sizeof(double) * totalS);
        }

        int minX = cvCeil(z[0] - m_params.innerRadX),
            maxX = cvFloor(z[0] + m_params.innerRadX),
            minY = cvCeil(z[1] - m_params.innerRadY),
            maxY = cvFloor(z[1] + m_params.innerRadY);

        OutputReal innerRadSq = m_params.innerRadX * m_params.innerRadX;

        constexpr OutputReal eps = (OutputReal)1e-4;
        const size_t nSigs = m_params.numSignals, ystep = m_pixels.stepT();
        auto pixLine = m_pixels[minY];
        auto sigs = z + 2;

//        XPRINTZ("add point: (%.3f;%.3f;%.3f); min: [%d;%d]; max: [%d;%d] -- %d %d",
//                        z[0], z[1], z[2], minX, minY, maxX, maxY, ptrMin, ptrMax);

        for(int iy = minY; iy <= maxY; iy++, pixLine += ystep)
        {
            if((uint32_t)iy >= m_params.h)
                continue;

            OutputReal dx = minX - (OutputReal)z[0], dy = iy - (OutputReal)z[1], dyQ = dy * dy;

            auto ppix = pixLine + minX * nSigs;
            for (auto ix = minX; ix <= maxX; ix++, dx += 1)
            {
                OutputReal wd = dyQ + dx * dx;
                if((uint32_t)ix >= m_params.w /*|| wd > innerRadSq*/) {
                    ppix += nSigs;
                    continue;
                }

                OutputReal denom = (wd < eps && false ? OutputReal(-1) : (OutputReal)1 / wd);
                for (size_t i = 0; i < nSigs; i++, ppix++)
                {
                    // this point is either not set (NaN) or has been set exactly..
                    //if (!std::isfinite(sigs[i]) || ppix->denom < 0)
                    //    continue;

                    denom = 1;
                    ppix->num += (OutputReal)sigs[i] * denom, ppix->denom += denom;
                    ppix->Ninner = 1;
                    continue; ///! HACK

                    ppix->Ninner++;
                    if (denom < 0)
                    {                                             // set this point exactly
                        ppix->num = -(OutputReal)sigs[i], ppix->denom = -1; // negative denom indicates the exact point
                    }
                    else
                    {
                        ppix->num += (OutputReal)sigs[i] / wd, ppix->denom += denom;
                    }
                } // for signals
            }     // for ix
        }         // for iy
    }

    void processOuterPoint(const double *z) noexcept
    {
        int minX = std::max(0, cvCeil(z[0] - m_params.outerRadX)),
            maxX = std::min((int)m_params.w - 1, cvFloor(z[0] + m_params.outerRadX)),
            minY = std::max(0, cvCeil(z[1] - m_params.outerRadY)),
            maxY = std::min((int)m_params.h - 1, cvFloor(z[1] + m_params.outerRadY));

        OutputReal innerRadSq = m_params.innerRadX * m_params.innerRadX,
                    outerRadSq = m_params.outerRadX * m_params.outerRadX;
        constexpr OutputReal PI = OutputReal(M_PI), PI2 = OutputReal(M_PI * 2);

        const size_t nSigs = m_params.numSignals, ystep = m_pixels.stepT(),
                     ptrMin = minX * nSigs, ptrMax = maxX * nSigs;
        auto ptr = m_pixels[minY];
        auto sigs = z + 2;

        for (int iy = minY; iy <= maxY; iy++, ptr += ystep)
        {
            OutputReal dx = minX - (OutputReal)z[0], dy = iy - (OutputReal)z[1], dyQ = dy * dy;

            for (auto ix = ptr + ptrMin; ix <= ptr + ptrMax; dx += 1)
            {
                OutputReal wd = dyQ + dx * dx;
                if (!(innerRadSq < wd && wd <= outerRadSq))
                {
                    ix += nSigs;
                    continue;
                }

                OutputReal angVal = 0, denom = -1;
                for (size_t i = 0; i < nSigs; i++, ix++)
                {
                    // this point is either not set (NaN) or has been set exactly..
                    if (ix->Ninner > 0 || !std::isfinite(sigs[i]))
                    {
                        //       m_innerPts++;
                        continue;
                    }

                    if (denom < 0)
                    { // compute it only once on demand..
                        angVal = std::atan2(dy, dx) + PI;
                        denom = OutputReal(1) / wd;
                    }
                    ix->num += (OutputReal)sigs[i] * denom, ix->denom += denom;
                    ix->Nouter++;
                    //                    m_outerPts++;

                    if (ix->Nouter == 1)
                    {
                        ix->angMin = angVal, ix->angMax = angVal;
                    }
                    else
                    {
                        auto d1 = ix->angMin - angVal,
                             d2 = angVal - ix->angMax;

                        if (ix->angMin <= ix->angMax)
                        {
                            if (d1 <= 0 && d2 <= 0) // this angle is inside
                                continue;
                        }
                        else
                        {                           // sweep over 0
                            if (d1 <= 0 || d2 <= 0) // this angle is inside
                                continue;
                        }
                        if (d1 < 0)
                            d1 += PI2;
                        if (d2 < 0)
                            d2 += PI2;
                        if (d1 < d2)
                            ix->angMin = angVal;
                        else
                            ix->angMax = angVal;
                    }
                } // for signals
            }     // for ix
        }         // for iy
    }

    bool process(std::atomic_bool &cancel)
    {
        if (!m_isReset)
            throw std::runtime_error("DataInterpolator must be reinitialized before processing!");

        m_isReset = false;

        //        XPRINTZ("Processing outer points: %d; out: %p; %d x %d; signals: %d; rad: [%f %f] - [%f %f]",
        //                m_scanPts.size(), m_outPtr,
        //                m_params.w, m_params.h, m_params.numSignals,
        //                m_params.innerRadX, m_params.innerRadY,
        //                m_params.outerRadX, m_params.outerRadY);

        const size_t totalS = m_params.numSignals + 2;
        for (auto ptr = m_scanPts.begin(); ptr != m_scanPts.end(); ptr += totalS)
        {
            //processOuterPoint(&*ptr);
            if (cancel.load())
            {
                return false;
            }
        }
//        XPRINTZ("# outer pt: %d; inner: %d", m_outerPts, m_innerPts);

        constexpr auto nan = std::numeric_limits<OutputReal>::quiet_NaN();
        const auto nSigs = m_params.numSignals, imgStep = m_params.w * nSigs;
        auto pimg = m_outPtr;

        for (int y = 0; y < (int)m_params.h; y++, pimg += imgStep)
        {
            if (cancel.load())
            {
                return false;
            }

            auto pix = m_pixels[y];
            auto pimgX = pimg;
            for (int x = 0; x < (int)m_params.w; x++, pimgX += nSigs)
            {
                for (size_t i = 0; i < nSigs; i++, pix++)
                {
                    if (pix->Ninner >= 1)
                    {
                        pimgX[i] = pix->num / pix->denom;
                    }
                    else if (pix->Nouter >= 2)
                    {
                        auto d = pix->angMax - pix->angMin;
                        d += (OutputReal)(d < 0 ? 2 * M_PI : 0.0); // sweep over 0
                        pimgX[i] = d < minAngleRange ? nan : pix->num / pix->denom;
                    }
                    else
                        pimgX[i] = nan;
                } // for signals
            }     // for x
        }         // for y
        return true;
    }

    void saveToFile(const std::string &outBinFile)
    {
        m_pofs.reset(new std::ofstream(outBinFile, std::ios::binary | std::ios::trunc));
        if (!m_pofs->is_open())
            throw std::runtime_error("Unable to open: " + outBinFile + " for writing!");

        m_pofs->write((const char *)&m_params, sizeof(m_params));
    }

    std::tuple< size_t, std::vector< InputReal > > readFromFile(const std::string &inBinFile)
    {
        std::ifstream ifs(inBinFile, std::ios::binary);
        ifs.exceptions(std::ifstream::badbit /*| std::ifstream::failbit*/);

        if (!ifs.is_open())
        {
            throw std::runtime_error("Unable to open: " + inBinFile + " for reading!");
        }

        Params params = {};
        ifs.read((char *)&params, sizeof(params));
        XPRINTZ("image sizes: %d x %d x %d", params.w, params.h, params.numSignals);

        //if (!(params.w == m_params.w && params.h == m_params.h && params.numSignals == m_params.numSignals))
        if (!(params.numSignals == m_params.numSignals))
        {
            throw std::runtime_error("Input buffer dimensions do not match: " +
                                     std::to_string(params.w) + "x" + std::to_string(params.h) +
                                     "x" + std::to_string(params.numSignals));
        }

        std::vector< InputReal > out, pt(2 + params.numSignals);
        out.reserve(params.w * params.h * pt.size());
        size_t nSamples = 0;
        for(; ifs; nSamples++)
        {
            ifs.read((char *)pt.data(), sizeof(InputReal) * pt.size());
            out.insert(out.end(), begin(pt), end(pt));
        }
        XPRINTZ("read total %u data samples", nSamples);

        return {nSamples, out};
    }
};

#endif // INTERPOLATE_HOST_H
