#include "stdio.h"
#include "zenoh.hxx"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using namespace zenoh;
using namespace std;
using namespace cv;

// 灯条结构体，使用RotatedRect
struct LightBar {
    RotatedRect rect;
    Point2f center;
    float length;    //灯条长边
    float width;     //灯条短边
    float angle;
    string color;    //red / blue
};

// GUI数据包，只保留结果图
struct GuiPack{
    Mat result;
};

vector<LightBar> detectLightBars(const Mat& srcBGR) {
    Mat gray, binary;
    cvtColor(srcBGR, gray, COLOR_BGR2GRAY);
    // 对比度增强
    gray.convertTo(gray, -1, 1.8, 8);
    GaussianBlur(gray, gray, Size(3, 3), 0);
    // Otsu阈值
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2,2));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<LightBar> lights;

    auto extractLights = [&](const vector<vector<Point>>& cnts,
                             vector<LightBar>& out,
                             bool strict_mode)
    {
        for (auto& cnt : cnts) {
            double area = contourArea(cnt);
            float min_area, min_len, ratio_min, ratio_max, max_angle;
            if(strict_mode)
            {
                min_area = 5;
                min_len = 5;
                ratio_min = 1.2;
                ratio_max =25;
                max_angle =40;
            }
            else
            {
                min_area = 3;
                min_len = 3;
                ratio_min =1.05;
                ratio_max =30;
                max_angle =50;
            }

            if (area < min_area) continue;
            if(cnt.size() < 6) continue;

            RotatedRect rrect = minAreaRect(cnt);
            float w = rrect.size.width;
            float h = rrect.size.height;

            float len = max(w, h);
            float wid = min(w, h);

            // 角度换算，修复竖直灯条bug
            float raw_angle = rrect.angle;
            float ang;
            if (w >= h)
            {
                ang = fabs(raw_angle + 90.0f);
            }
            else
            {
                ang = fabs(raw_angle);
            }
            if (ang > 45.0f)
            {
                ang = 90.0f - ang;
            }

            if(len < min_len) continue;
            float ratio = len / wid;
            if (ratio < ratio_min || ratio > ratio_max) continue;
            if (ang > max_angle) continue;

            Point c_pt(cvRound(rrect.center.x), cvRound(rrect.center.y));
            int half = 2;
            Rect roi(c_pt.x-half, c_pt.y-half, half*2+1, half*2+1);
            roi &= Rect(0,0,srcBGR.cols, srcBGR.rows);
            Mat patch = srcBGR(roi);
            Scalar avg = mean(patch);
            double B = avg[0];
            double R = avg[2];
            string col = (B > R) ? "blue" : "red";

            LightBar lb;
            lb.rect = rrect;
            lb.center = rrect.center;
            lb.length = len;
            lb.width = wid;
            lb.angle = ang;
            lb.color = col;
            out.push_back(lb);
        }
    };

    //第一轮严格模式
    extractLights(contours, lights, true);

    // Otsu无灯条，降级固定阈值
    if(lights.empty())
    {
        threshold(gray, binary, 90, 255, THRESH_BINARY);
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        contours.clear();
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        extractLights(contours, lights, true);
    }

    //灯条小于2，切换宽松筛选
    if(lights.size() < 2)
    {
        lights.clear();
        extractLights(contours, lights, false);
    }
    return lights;
}

//灯条配对，绘制装甲框 + 装甲中心点(红色实心圆点+文字)
vector<vector<Point2f>> matchArmor(const vector<LightBar>& lights, Mat& drawImg) {
    vector<vector<Point2f>> armors;
    for (int i = 0; i < lights.size(); i++) {
        for (int j = i+1; j < lights.size(); j++) {
            const LightBar& l1 = lights[i];
            const LightBar& l2 = lights[j];
            if(l1.color != l2.color) continue;

            float longer = max(l1.length, l2.length);
            float shorter = min(l1.length, l2.length);
            float lenRatio = longer / shorter;
            if(lenRatio > 1.5) continue;

            float cx1=l1.center.x, cy1=l1.center.y;
            float cx2=l2.center.x, cy2=l2.center.y;
            float dist = sqrt(pow(cx1-cx2,2)+pow(cy1-cy2,2));
            float meanLen = (l1.length + l2.length)/2;
            float distRatio = dist / meanLen;
            if(distRatio < 0.8 || distRatio >6) continue;

            auto getEndPoints = [](RotatedRect r)->pair<Point2f,Point2f>{
                Point2f pts[4];
                r.points(pts);
                sort(pts, pts+4, [](Point2f a,Point2f b){return a.y < b.y;});
                Point2f top = (pts[0]+pts[1])/2;
                Point2f bot = (pts[2]+pts[3])/2;
                return {top, bot};
            };
            auto [t1,b1] = getEndPoints(l1.rect);
            auto [t2,b2] = getEndPoints(l2.rect);

            vector<Point2f> armorPts = {t1, t2, b2, b1};
            armors.push_back(armorPts);

            //绘制绿色装甲四边形
            for(int p=0;p<4;p++){
                line(drawImg, armorPts[p], armorPts[(p+1)%4], Scalar(0,255,0),2);
            }

            //计算装甲中心点
            Point2f armor_center(
                (armorPts[0].x + armorPts[1].x + armorPts[2].x + armorPts[3].x)/4.0f,
                (armorPts[0].y + armorPts[1].y + armorPts[2].y + armorPts[3].y)/4.0f
            );
            //画红色实心圆点
            circle(drawImg, armor_center, 4, Scalar(0,0,255), -1);
            putText(drawImg, "center", armor_center + Point2f(8,-6),
                    FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,0,255),1);

            putText(drawImg, "candidate:" + l1.color, t1-Point2f(0,10),
                    FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,0),1);
        }
    }
    return armors;
}

struct FrameData {
    cv::Mat mat;
    uint64_t seq;
    uint64_t ts;
};

queue<FrameData> frame_queue;
mutex mtx;
condition_variable cv_cond;
bool running = true;
const size_t MAX_QUEUE_SIZE = 2;

mutex frame_mtx;
queue<GuiPack> frame_q;
const size_t MAX_GUI_QUEUE = 1;

void worker_thread() {
    while (running) {
        unique_lock<mutex> lock(mtx);
        cv_cond.wait(lock, []() { return !frame_queue.empty() || !running; });
        if (!running) break;
        FrameData data = frame_queue.front();
        frame_queue.pop();
        lock.unlock();

        Mat frame = data.mat.clone();
        vector<LightBar> lights = detectLightBars(frame);

        //绘制黄色灯条框
        for(int i=0;i<lights.size();i++){
            Point2f pts[4];
            lights[i].rect.points(pts);
            for(int p=0;p<4;p++){
                line(frame, pts[p], pts[(p+1)%4], Scalar(0,255,255), 2);
            }
            putText(frame, "L"+to_string(i)+" "+lights[i].color,
                    lights[i].center+Point2f(8,0),
                    FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        }

        auto armorList = matchArmor(lights, frame);
        cout << "seq:" << data.seq << " lights:" << lights.size() << " armors:" << armorList.size() << endl;

        GuiPack pack;
        pack.result = frame;
        {
            lock_guard<mutex> lk(frame_mtx);
            while(frame_q.size() >= MAX_GUI_QUEUE) frame_q.pop();
            frame_q.push(pack);
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        string key = argc > 1 ? argv[1] : "training/demo/status";
        auto config = Config::create_default();
        config.insert_json5("transport/shared_memory/enabled", "true");
        auto session = Session::open(move(config));

        auto on_data = [](const Sample &sample) {
            auto&&payload = sample.get_payload();
            std::vector<uint8_t> bytes_view = payload.as_vector();
            const uint8_t *buf = bytes_view.data();
            size_t buf_len = bytes_view.size();
            if (buf_len < 24) {
                cerr << "small packet drop\n";
                return;
            }
            uint64_t seq = 0;
            uint64_t ts_ns = 0;
            uint32_t w = 0;
            uint32_t h = 0;
            memcpy(&seq, buf + 0, sizeof(uint64_t));
            memcpy(&ts_ns, buf + 8, sizeof(uint64_t));
            memcpy(&w, buf + 16, sizeof(uint32_t));
            memcpy(&h, buf + 20, sizeof(uint32_t));
            size_t pixel_bytes = (size_t)w * h * 3;
            if (buf_len != 24 + pixel_bytes) {
                cerr << "size mismatch drop seq=" << seq << "\n";
                return;
            }
            Mat frame(h, w, CV_8UC3, (void *)(buf + 24));
            FrameData fd;
            fd.mat = frame.clone();
            fd.seq = seq;
            fd.ts = ts_ns;

            lock_guard<mutex> lock(mtx);
            while(frame_queue.size() >= MAX_QUEUE_SIZE) frame_queue.pop();
            frame_queue.push(fd);
            cv_cond.notify_one();
        };

        auto subscriber = session.declare_subscriber(KeyExpr(key), on_data, closures::none);
        cout << "Subscribed, press ESC to quit\n";
        thread worker(worker_thread);

        namedWindow("result", WINDOW_NORMAL);
        GuiPack cache;
        while (true) {
            GuiPack disp_pack;
            {
                std::lock_guard<std::mutex> lock(frame_mtx);
                if (!frame_q.empty()) {
                    disp_pack = frame_q.front();
                    frame_q.pop();
                }
            }
            if(!disp_pack.result.empty()){
                cache = disp_pack;
            }
            if (!cache.result.empty()) {
                imshow("result", cache.result);
            }

            int key = waitKey(1);
            if(key == 27)
            {
                running = false;
                cv_cond.notify_all();
                break;
            }
        }
        worker.join();
        session.close();
    } catch (const exception &error) {
        cerr << "sub error:" << error.what() << endl;
        return 1;
    }
    return 0;
}
