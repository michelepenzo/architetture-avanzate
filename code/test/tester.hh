#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <string>
#define TESTER_SMALL_INSTANCES_MIN 16
#define TESTER_SMALL_INSTANCES_MAX 17
#define TESTER_BIG_INSTANCES_MIN 20'000
#define TESTER_BIG_INSTANCES_MAX -1

class tester {

    const std::string name;

    virtual bool test_instance(int instance_number) = 0;

    bool test_instances(size_t from, size_t to, bool is_exp = false) {
        
        bool all_ok = true;

        for(int m = from; m < to; m = is_exp ? m*1.5 : m+1) {
            std::cout << name << " - testing m=" << std::setw(10) << m << ": " << std::flush;
            bool ok = test_instance(m);
            std::cout << (ok ? "OK" : "NO") << std::endl << std::flush;
            all_ok &= ok;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        return all_ok;
    }

public:

    tester(std::string name) : name(name) { }

    bool test_many_instances() {
        
        bool all_ok = true;

        all_ok &= test_instances(TESTER_SMALL_INSTANCES_MIN, TESTER_SMALL_INSTANCES_MAX);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        all_ok &= test_instances(TESTER_BIG_INSTANCES_MIN, TESTER_BIG_INSTANCES_MAX, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return all_ok;
    }
};
