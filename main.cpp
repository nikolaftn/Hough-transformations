#include <iostream>
#include <string>
#include <memory>
#include "Image.h"
#include <fstream>
#include <tbb/flow_graph.h>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace tbb::flow;

struct ImageTask
{
    string inputPath;
    string outputPath;
    shared_ptr<Image> image;

    bool parallelVersion;

    double loadTime;
    double grayTime;
    double sobelTime;
    double houghTime;
    double detectTime;
    double drawSaveTime;
};

void ProcessImagesFlowGraph(const vector<string>& paths, bool parallelVersion)
{
    graph g;

    function_node<string, ImageTask> loadNode(g, unlimited,
        [parallelVersion](const string& path)
        {
            auto start = chrono::high_resolution_clock::now();

            ImageTask task;
            task.inputPath = path;
            task.parallelVersion = parallelVersion;

            string imageName =
                filesystem::path(path).stem().string();

            string resultFolder;

            if (parallelVersion)
            {
                resultFolder =
                    string(R"(C:\Users\Nikola\Desktop\rezultatiparallel\)") +
                    imageName;
            }
            else
            {
                resultFolder =
                    string(R"(C:\Users\Nikola\Desktop\rezultatiserial\)") +
                    imageName;
            }

            filesystem::create_directories(resultFolder);

            task.outputPath =
                resultFolder + R"(\output.png)";

            task.image = make_shared<Image>();
            task.image->Load(path);

            auto end = chrono::high_resolution_clock::now();
            task.loadTime =
                chrono::duration<double, milli>(end - start).count();

            return task;
        });

    function_node<ImageTask, ImageTask> grayNode(g, unlimited,
        [](ImageTask task)
        {
            auto start = chrono::high_resolution_clock::now();

            if (task.parallelVersion)
                task.image->ConvertToGrayscaleParallel();
            else
                task.image->ConvertToGrayscale();

            auto end = chrono::high_resolution_clock::now();
            task.grayTime = chrono::duration<double, milli>(end - start).count();

            return task;
        });

    function_node<ImageTask, ImageTask> sobelNode(g, unlimited,
        [](ImageTask task)
        {
            auto start = chrono::high_resolution_clock::now();

            if (task.parallelVersion)
                task.image->DetectEdgesSobelParallel(300);
            else
                task.image->DetectEdgesSobel(300);

            task.image->SaveEdges(
                filesystem::path(task.outputPath).parent_path().string()
                + "\\edges.png"
            );

            auto end = chrono::high_resolution_clock::now();
            task.sobelTime = chrono::duration<double, milli>(end - start).count();

            return task;
        });

    function_node<ImageTask, ImageTask> houghNode(g, unlimited,
        [](ImageTask task)
        {
            auto start = chrono::high_resolution_clock::now();

            if (task.parallelVersion)
                task.image->ComputeHoughTransformParallel();
            else
                task.image->ComputeHoughTransform();

            auto end = chrono::high_resolution_clock::now();
            task.houghTime = chrono::duration<double, milli>(end - start).count();

            return task;
        });

    function_node<ImageTask, ImageTask> detectNode(g, unlimited,
        [](ImageTask task)
        {
            auto start = chrono::high_resolution_clock::now();

            if (task.parallelVersion)
                task.image->DetectLinesParallel(100);
            else
                task.image->DetectLines(100);

            auto end = chrono::high_resolution_clock::now();
            task.detectTime = chrono::duration<double, milli>(end - start).count();

            return task;
        });

    function_node<ImageTask> saveNode(g, unlimited,
        [](ImageTask task)
        {
            auto start = chrono::high_resolution_clock::now();

            task.image->DrawDetectedLines(task.parallelVersion);
            task.image->Save(task.outputPath);

            auto end = chrono::high_resolution_clock::now();

            task.drawSaveTime =
                chrono::duration<double, milli>(end - start).count();

            ofstream file(
                filesystem::path(task.outputPath).parent_path()
                / "result.txt"
            );

            file << "Image: " << task.inputPath << "\n";
            file << "Load: " << task.loadTime << " ms\n";
            file << "Grayscale: " << task.grayTime << " ms\n";
            file << "Sobel: " << task.sobelTime << " ms\n";
            file << "Hough Transform: " << task.houghTime << " ms\n";
            file << "Line Detection: " << task.detectTime << " ms\n";
            file << "Drawing and Saving: " << task.drawSaveTime << " ms\n";
            file << "Detected lines: "
                << task.image->GetDetectedLinesCount(task.parallelVersion)
                << "\n";

            if (task.parallelVersion)
                cout << "\n========== PARALLEL ==========\n";
            else
                cout << "\n========== SERIAL ==========\n";

            cout << "Image: " << task.inputPath << endl;
            cout << "Load: " << task.loadTime << " ms\n";
            cout << "Grayscale: " << task.grayTime << " ms\n";
            cout << "Sobel: " << task.sobelTime << " ms\n";
            cout << "Hough Transform: " << task.houghTime << " ms\n";
            cout << "Line Detection: " << task.detectTime << " ms\n";
            cout << "Drawing and Saving: " << task.drawSaveTime << " ms\n";
            cout << "Detected lines: "
                << task.image->GetDetectedLinesCount(task.parallelVersion)
                << endl;
        });

    make_edge(loadNode, grayNode);
    make_edge(grayNode, sobelNode);
    make_edge(sobelNode, houghNode);
    make_edge(houghNode, detectNode);
    make_edge(detectNode, saveNode);

    for (const auto& path : paths)
        loadNode.try_put(path);

    g.wait_for_all();
}

int main()
{
    while (true)
    {
        cout << "===== MENU =====\n";
        cout << "1. Start Processing\n";
        cout << "q - Exit\n";
        cout << "Enter your choice: \n";

        string choice;
        getline(cin, choice);

        if (choice == "1")
        {
            vector<string> paths;

            for (const auto& entry : filesystem::directory_iterator(R"(C:\Users\Nikola\Desktop\slikepp)"))
            {
                if (entry.is_regular_file())
                {
                    paths.push_back(entry.path().string());
                }
            }

            cout << "\n====================================\n";
            cout << "STARTING SERIAL PROCESSING\n";
            cout << "Number of images: " << paths.size() << endl;
            cout << "====================================\n";

            auto start = chrono::high_resolution_clock::now();

            ProcessImagesFlowGraph(paths, false);

            auto end = chrono::high_resolution_clock::now();

            double serialTime =
                chrono::duration<double, milli>(end - start).count();

            cout << "\n====================================\n";
            cout << "SERIAL PROCESSING COMPLETED\n";
            cout << "Total time: " << serialTime << " ms\n";
            cout << "====================================\n";

            cout << "\n====================================\n";
            cout << "STARTING PARALLEL PROCESSING\n";
            cout << "Number of images: " << paths.size() << endl;
            cout << "====================================\n";

            start = chrono::high_resolution_clock::now();

            ProcessImagesFlowGraph(paths, true);

            end = chrono::high_resolution_clock::now();

            double parallelTime =
                chrono::duration<double, milli>(end - start).count();

            cout << "\n====================================\n";
            cout << "PARALLEL PROCESSING COMPLETED\n";
            cout << "Total time: " << parallelTime << " ms\n";
            cout << "====================================\n";

            double speedup = serialTime / parallelTime;

            cout << "\n====================================\n";
            cout << "FINAL RESULTS\n";
            cout << "====================================\n";
            cout << "Serial execution time: " << serialTime << " ms\n";
            cout << "Parallel execution time: " << parallelTime << " ms\n";
            cout << "Speedup: " << speedup << endl;

            ofstream file(
                R"(C:\Users\Nikola\Desktop\comparison_results.txt)"
            );

            file << "Number of images: " << paths.size() << "\n";
            file << "Total serial execution time: "
                << serialTime
                << " ms\n";

            file << "Total parallel execution time: "
                << parallelTime
                << " ms\n";

            file << "Speedup: "
                << speedup
                << "\n";
        }
        else if (choice == "q")
        {
            return 0;
        }
        else
        {
            cout << "Invalid option. Please try again.\n";
        }
    }
}
