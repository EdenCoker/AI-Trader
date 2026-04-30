#include <iostream>

#include "alpha_bridge.hpp"

int main() {
    ai_trader::AlphaBridgeReader reader;
    while (true) {
        auto msg = reader.poll(5000);
        if (!msg) {
            std::cout << "timeout\n";
            continue;
        }
        std::cout << msg->ticker << " direction=" << static_cast<int>(msg->direction)
                  << " conviction=" << msg->conviction
                  << " size=" << msg->size_multiplier
                  << " holding_days=" << msg->holding_period_days
                  << " exit=" << msg->exit_trigger << "\n";
    }
}

