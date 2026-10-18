#include "zenoh.hxx"
#include <chrono>
#include <exception>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <utility>
int main(int argc, char *argv[]) {
  using namespace zenoh;
  try {
    std::string key = argc > 1 ? argv[1] : "training/demo/status";
    auto config = Config::create_default();
    config.insert_json5("transport/shared_memory/enabled", "true");
    auto session = Session::open(std::move(config));
    auto publisher = session.declare_publisher(KeyExpr(key));

    std::cout << "Publishing " << key << ". Ctrl+C to stop." << std::endl;
    uint64_t seq = 0;
    cv::VideoCapture video("raw.mp4");
    if (!video.isOpened()) {
      std::cerr << "Cannot open video" << std::endl;
      return 1;
    }
    cv::Mat frame;
    while (video.read(frame)) {
      int w = frame.cols;
      int h = frame.rows;
      uint64_t ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();

      std::vector<uint8_t> packet;
      packet.resize(24 + (size_t)w * h * 3);

      memcpy(packet.data() + 0, &seq, sizeof(uint64_t));
      memcpy(packet.data() + 8, &ts_ns, sizeof(uint64_t));
      memcpy(packet.data() + 16, &w, sizeof(uint32_t));
      memcpy(packet.data() + 20, &h, sizeof(uint32_t));
      memcpy(packet.data() + 24, frame.data, (size_t)w * h * 3);

      publisher.put(packet);
      std::cout << "send seq:" << seq << "  " << w << "x" << h << std::endl;
      seq++;
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
      if (cv::waitKey(33) == 27)
        break;
    }
  } catch (const std::exception &error) {
    std::cerr << "pub: " << error.what() << std::endl;
    return 1;
  }
}
