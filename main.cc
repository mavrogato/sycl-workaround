
#include <iostream>
#include <vector>
#include <complex>

#include <CL/sycl.hpp>

int main() {
    try {
        auto selector = sycl::gpu_selector{};
        {
            auto dev = selector.select_device();
            std::cout << "# gpu device found: ";
            std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
        }
        static constexpr size_t N = 64 * 1024;
        std::vector<std::complex<float>> v(N);
        {
            auto buf = sycl::buffer{v};
            sycl::queue{selector}.submit([&buf](sycl::handler& h) {
                auto a = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::range<1>{N}, [a](sycl::id<1> idx) {
                    static constexpr auto PI = 3.141592653589793;
                    static constexpr auto PS = 0.618033988749895;
                    static constexpr auto AL = 2 * PI * PS;
                    double i = idx;
                    auto tmp = std::pow(std::polar(1.0, AL), i) * std::sqrt(i + 1.0);
                    a[idx] = std::complex<float>(tmp.real(), tmp.imag());
                });
            });
        }
        for (auto z : v) {
            std::cout << z.real() << ',' << z.imag() << std::endl;
        }
    }
    catch (...) {
        std::cerr << "something wrong..." << std::endl;
        std::rethrow_exception(std::current_exception());
    }
    return 0;
}
