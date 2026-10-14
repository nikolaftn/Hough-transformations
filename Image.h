#pragma once
using namespace std;

#include <vector>
#include <string>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <map>

struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct DetectedLine
{
    int rho;
    int theta;
    int votes;
};

struct HoughKey
{
    int rho;
    int theta;

    bool operator==(const HoughKey& other) const
    {
        return rho == other.rho &&
            theta == other.theta;
    }

    bool operator<(const HoughKey& other) const //Za obicnu mapu
    {
        if (rho != other.rho)
            return rho < other.rho;

        return theta < other.theta;
    }
};

struct HoughHashCompare
{
    static size_t hash(const HoughKey& key)
    {
        std::hash<int> hasher;

        size_t rhoHash = hasher(key.rho);
        size_t thetaHash = hasher(key.theta);

        return rhoHash * 31 + thetaHash;
    }

    static bool equal(const HoughKey& a, const HoughKey& b)
    {
        return a == b;
    }
};

class Image
{
private:
    int width;
    int height;
    vector<Pixel> originalPixels; //Da se sacuva original
    vector<Pixel> pixels;
    vector<unsigned char> edgePixels;    //Kontejner sadrzi char jer nema potrebe da sadrzi Pixel koji ima r,g,b
    tbb::concurrent_hash_map<HoughKey, int, HoughHashCompare> accumulator;    //key je (rho, teta a value ce biti broj glasova za tu pravu)
    tbb::concurrent_vector<DetectedLine> detectedLines; //Vektor za detektovane linije nakon Hough transformacije
    map<HoughKey, int> serialAccumulator;
    vector<DetectedLine> serialDetectedLines;

public:
    Image();

    int GetWidth() const;
    int GetHeight() const;

    void SetWidth(int width);
    void SetHeight(int height);
    vector<Pixel>& GetPixels();
    vector<unsigned char>& GetEdgePixels();
    int GetDetectedLinesCount(bool parallelVersion) const;

    bool Load(string path);
    bool Save(string path);

    // Paralelne verzije
    void ConvertToGrayscaleParallel();
    void DetectEdgesSobelParallel(double theshold);
    void ComputeHoughTransformParallel();
    void DetectLinesParallel(int voteThreshold);

    // Serijske verzije
    void ConvertToGrayscale();
    void DetectEdgesSobel(double threshold);
    void ComputeHoughTransform();
    void DetectLines(int voteThreshold);

    void DrawDetectedLines(bool);
    void DrawLine(int x1, int y1, int x2, int y2);

    bool SaveEdges(string path);

};

