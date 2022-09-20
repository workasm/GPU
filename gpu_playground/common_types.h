
#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#include <stdint.h>
#include <vector>

template<class Real>
struct InterpPix
{
    Real num,            // numerator and
        denom;           // denominator for Inverse distance weighting
    Real angMin, angMax; // minimal ang maximal connection angle for outer points
    uint16_t Ninner;     // # of inner points collected
    uint16_t Nouter;     // # of outer points collected
};

template < class Real >
struct InterpParams
{
    uint32_t w, h; // original image size in pixels
    uint32_t numSignals;
    Real innerRadX, innerRadY; // radius of the inner ellipse in pixel fractions
    Real outerRadX, outerRadY; // radius of the outer ellipse in pixel fractions
};

template <typename T>
class TImage : std::vector< T > {

    enum {
        s_align = 4,
        s_ofs = (1 << s_align) - 1,
    };

    using Base = std::vector< T >;
public:
    using NT = T;
    using Base::data;
    using Base::size;
    using Base::begin;
    using Base::end;

    explicit TImage(size_t w = 0, size_t h = 0) {
        resize(w, h);
    }

    TImage(size_t w, size_t h, const T& value) {
        resize(w, h, value);
    }

    TImage(const T *ptr, size_t w, size_t h) :
        TImage(w, h)
    {
        std::copy(ptr, ptr + size(), data());
    }

    void resize(size_t w, size_t h) {
        m_width = w, m_height = h, m_stride = (w + s_ofs) & ~s_ofs;
        Base::resize(m_stride * h);
    }

    void resize(size_t w, size_t h, const T& value) {
        m_width = w, m_height = h, m_stride = (w + s_ofs) & ~s_ofs;
        Base::resize(m_stride * h, value);
    }

    void swap(TImage& rhs) {
        std::swap(m_width, rhs.m_width);
        std::swap(m_height, rhs.m_height);
        std::swap(m_stride, rhs.m_stride);
        Base::swap(rhs);
    }

    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }

    const T* operator[](size_t i) const { return data() + i * m_stride; }
    T* operator[](size_t i)	{ return data() + i * m_stride; }

    size_t stepT() 	const { return m_stride; }          // line size in type T
    //size_t step() 	const { return stepT()*sizeof(T); } // line size in bytes

    size_t width() 	const { return m_width; }
    size_t height()	const { return m_height; }

    size_t sizeInBytes() const { return size() * sizeof(T); }

private:
     size_t m_width, m_height, m_stride;
};

#endif // COMMON_TYPES_H_
