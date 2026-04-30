#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace ai_trader {

#pragma pack(push, 1)
struct TradePlanMsg {
    std::uint64_t timestamp_ns;
    char ticker[16];
    std::int8_t direction;
    char pad[3];
    float conviction;
    float size_multiplier;
    std::int32_t holding_period_days;
    char exit_trigger[128];
    std::uint8_t checksum;
    char pad2[3];
};
#pragma pack(pop)

static_assert(sizeof(TradePlanMsg) == 172, "TradePlanMsg must be 172 bytes");

inline std::uint8_t checksum(const TradePlanMsg& msg) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&msg);
    std::uint8_t value = 0;
    for (std::size_t i = 0; i < 168; ++i) {
        value ^= bytes[i];
    }
    return value;
}

class AlphaBridgeReader {
public:
    AlphaBridgeReader() {
        shm_fd_ = shm_open("/ai_trader_bridge", O_CREAT | O_RDWR, 0600);
        if (shm_fd_ < 0) {
            throw std::runtime_error("shm_open failed");
        }
        if (ftruncate(shm_fd_, sizeof(TradePlanMsg)) != 0) {
            throw std::runtime_error("ftruncate failed");
        }
        msg_ = static_cast<TradePlanMsg*>(
            mmap(nullptr, sizeof(TradePlanMsg), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0));
        if (msg_ == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        sem_ready_ = sem_open("/ai_trader_ready", O_CREAT, 0600, 0);
        sem_ack_ = sem_open("/ai_trader_ack", O_CREAT, 0600, 0);
        if (sem_ready_ == SEM_FAILED || sem_ack_ == SEM_FAILED) {
            throw std::runtime_error("sem_open failed");
        }
    }

    ~AlphaBridgeReader() {
        if (msg_ && msg_ != MAP_FAILED) {
            munmap(msg_, sizeof(TradePlanMsg));
        }
        if (shm_fd_ >= 0) {
            close(shm_fd_);
        }
        if (sem_ready_ && sem_ready_ != SEM_FAILED) {
            sem_close(sem_ready_);
        }
        if (sem_ack_ && sem_ack_ != SEM_FAILED) {
            sem_close(sem_ack_);
        }
    }

    std::optional<TradePlanMsg> poll(int timeout_ms = 5000) {
        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }
        if (sem_timedwait(sem_ready_, &ts) != 0) {
            return std::nullopt;
        }
        TradePlanMsg copy{};
        std::memcpy(&copy, msg_, sizeof(TradePlanMsg));
        sem_post(sem_ack_);
        if (copy.checksum != checksum(copy)) {
            return std::nullopt;
        }
        return copy;
    }

private:
    int shm_fd_ = -1;
    TradePlanMsg* msg_ = nullptr;
    sem_t* sem_ready_ = nullptr;
    sem_t* sem_ack_ = nullptr;
};

}  // namespace ai_trader

