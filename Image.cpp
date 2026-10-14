#include "Image.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <cmath>
#include <numbers>

Image::Image()
{
    width = 0;
    height = 0;
}

int Image::GetWidth() const
{
    return width;
}

int Image::GetHeight() const
{
    return height;
}

void Image::SetWidth(int width)
{
    this->width = width;
}

void Image::SetHeight(int height)
{
    this->height = height;
}

vector<Pixel>& Image::GetPixels()
{
    return pixels;
}

vector<unsigned char>& Image::GetEdgePixels() {
    return edgePixels;
}

int Image::GetDetectedLinesCount(bool parallelVersion) const
{
    if (parallelVersion)
        return detectedLines.size();

    return serialDetectedLines.size();
}

bool Image::Load(string path) {

    int channels;  //broj kanala (nijansi boja), postoji jer se nesto mora proslediti u narednoj funkciji na tom mestu

    unsigned char* data = stbi_load(  //stb funkcija za ucitavanje slike u niz karaktera
        path.c_str(),
        &width,
        &height,
        &channels,
        3
    );

    if (data == nullptr)
    {
        cout << "STB greska: "
            << stbi_failure_reason()
            << endl;

        return false;
    }

    pixels.resize(width * height); //osiguravam da vektor ima dovoljno mesta
    originalPixels.resize(width * height);

    for (int i = 0; i < width * height; i++) //ucitavanje svih piksela
    {
        pixels[i].r = data[3 * i];
        pixels[i].g = data[3 * i + 1];
        pixels[i].b = data[3 * i + 2];
    }
    stbi_image_free(data);   //Oslobadjanje memorije

    originalPixels = pixels;
    return true;

}

bool Image::Save(string path) {

    if (pixels.empty())
    {
        return false;
    }

    int result = stbi_write_png(
        path.c_str(),
        width,
        height,
        3,
        pixels.data(),
        width * 3         //stride in bytes (za png), 70 - 100 (za jpg) - kvalitet
    );

    return result != 0;
}

void Image::ConvertToGrayscaleParallel()
{
    tbb::parallel_for(
        tbb::blocked_range<int>(
            0,
            (int)pixels.size(),
            10000
        ),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int i = range.begin(); i != range.end(); ++i)
            {
                unsigned char gray =
                    0.299 * pixels[i].r +
                    0.587 * pixels[i].g +
                    0.114 * pixels[i].b;

                pixels[i] = { gray, gray, gray };
            }
        },
        tbb::auto_partitioner()
    );
}

void Image::ConvertToGrayscale()
{
    for (int i = 0; i < pixels.size(); i++)
    {
        unsigned char gray =
            0.299 * pixels[i].r +
            0.587 * pixels[i].g +
            0.114 * pixels[i].b;

        pixels[i] = { gray, gray, gray };
    }
}

void Image::DetectEdgesSobelParallel(double threshold)
{
    edgePixels.assign(width * height, 0);

    tbb::parallel_for(
        tbb::blocked_range<int>(1, height - 1, 32),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int y = range.begin(); y != range.end(); ++y)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    int index = y * width + x;

                    int a = pixels[(y - 1) * width + (x - 1)].r;
                    int b = pixels[(y - 1) * width + x].r;
                    int c = pixels[(y - 1) * width + (x + 1)].r;

                    int d = pixels[y * width + (x - 1)].r;
                    int f = pixels[y * width + (x + 1)].r;

                    int g = pixels[(y + 1) * width + (x - 1)].r;
                    int h = pixels[(y + 1) * width + x].r;
                    int i = pixels[(y + 1) * width + (x + 1)].r;

                    //koristi se maska
                    int gx =
                        -a + c
                        - 2 * d + 2 * f
                        - g + i;

                    int gy =
                        -a - 2 * b - c
                        + g + 2 * h + i;

                    double magnitude = sqrt(gx * gx + gy * gy);

                    if (magnitude > threshold)
                    {
                        edgePixels[index] = 0;
                    }
                    else
                    {
                        edgePixels[index] = 255;
                    }
                }
            }
        },
        tbb::auto_partitioner()
    );
}

void Image::DetectEdgesSobel(double threshold)
{
    edgePixels.assign(width * height, 0);

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int index = y * width + x;

            int a = pixels[(y - 1) * width + (x - 1)].r;
            int b = pixels[(y - 1) * width + x].r;
            int c = pixels[(y - 1) * width + (x + 1)].r;

            int d = pixels[y * width + (x - 1)].r;
            int f = pixels[y * width + (x + 1)].r;

            int g = pixels[(y + 1) * width + (x - 1)].r;
            int h = pixels[(y + 1) * width + x].r;
            int i = pixels[(y + 1) * width + (x + 1)].r;

            int gx =
                -a + c
                - 2 * d + 2 * f
                - g + i;

            int gy =
                -a - 2 * b - c
                + g + 2 * h + i;

            double magnitude = sqrt(gx * gx + gy * gy);

            if (magnitude > threshold)
            {
                edgePixels[index] = 0;
            }
            else
            {
                edgePixels[index] = 255;
            }
        }
    }
}

void Image::ComputeHoughTransformParallel()
{
    accumulator.clear();

    int maxRho = (int)sqrt(width * width + height * height);

    vector<double> cosTable(180);
    vector<double> sinTable(180);

    for (int theta = 0; theta < 180; theta++)
    {
        double thetaRad =
            theta * numbers::pi / 180.0;

        cosTable[theta] = cos(thetaRad);
        sinTable[theta] = sin(thetaRad);
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(0, height, 32),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int y = range.begin(); y != range.end(); ++y)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = y * width + x;

                    if (edgePixels[index] != 0) // ako nije ivica skip
                    {
                        continue;
                    }

                    for (int theta = 0; theta < 180; theta++)
                    {
                        int rho = (int)round(
                                x * cosTable[theta] +
                                y * sinTable[theta]
                            );

                        int rhoIndex = rho + maxRho;

                        HoughKey key
                        {
                            rhoIndex,
                            theta
                        };

                        tbb::concurrent_hash_map<
                            HoughKey,
                            int,
                            HoughHashCompare
                        >::accessor acc;

                        if (accumulator.insert(acc, key))
                        {
                            acc->second = 1;
                        }
                        else
                        {
                            acc->second++;
                        }
                    }
                }
            }
        },
        tbb::auto_partitioner()
    );
}

void Image::ComputeHoughTransform()
{
    serialAccumulator.clear();

    int maxRho = (int)sqrt(width * width + height * height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            if (edgePixels[index] != 0)
            {
                continue;
            }

            for (int theta = 0; theta < 180; theta++)
            {
                double thetaRad =
                    theta * numbers::pi / 180.0;

                int rho =
                    (int)(x * cos(thetaRad) +
                        y * sin(thetaRad));

                int rhoIndex = rho + maxRho;

                HoughKey key
                {
                    rhoIndex,
                    theta
                };

                serialAccumulator[key]++;
            }
        }
    }
}

void Image::DetectLinesParallel(int voteThreshold)
{
    detectedLines.clear();

    HoughKey neighborKey;

    tbb::parallel_for_each(
        accumulator.begin(),
        accumulator.end(),
        [&](const auto& entry)
        {
            const int rho = entry.first.rho;
            const int theta = entry.first.theta;
            const int votes = entry.second;

            if (votes < voteThreshold)
            {
                return;
            }

            bool isLocalMaximum = true;

            for (int dr = -1; dr <= 1 && isLocalMaximum; dr++)
            {
                for (int dt = -1; dt <= 1; dt++)
                {
                    if (dr == 0 && dt == 0)   //to je onaj element koji ispitujemo
                    {
                        continue;
                    }

                    neighborKey.rho = rho + dr;
                    neighborKey.theta = theta + dt;

                    tbb::concurrent_hash_map<
                        HoughKey,
                        int,
                        HoughHashCompare
                    >::const_accessor acc;

                    if (accumulator.find(acc, neighborKey) &&
                        acc->second >= votes)
                    {
                        isLocalMaximum = false;
                        break;
                    }
                }
            }

            if (isLocalMaximum)
            {
                detectedLines.push_back(
                    {
                        rho,
                        theta,
                        votes
                    });
            }
        });
}

void Image::DetectLines(int voteThreshold)
{
    serialDetectedLines.clear();

    for (const auto& entry : serialAccumulator)
    {
        int rho = entry.first.rho;
        int theta = entry.first.theta;
        int votes = entry.second;

        if (votes < voteThreshold)
        {
            continue;
        }

        bool isLocalMaximum = true;

        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dt = -1; dt <= 1; dt++)
            {
                if (dr == 0 && dt == 0)
                {
                    continue;
                }

                HoughKey neighborKey
                {
                    rho + dr,
                    theta + dt
                };

                auto it =
                    serialAccumulator.find(neighborKey);

                if (it != serialAccumulator.end())
                {
                    if (it->second >= votes)
                    {
                        isLocalMaximum = false;
                        break;
                    }
                }
            }

            if (!isLocalMaximum)
            {
                break;
            }
        }

        if (isLocalMaximum)
        {
            serialDetectedLines.push_back(
                {
                    rho,
                    theta,
                    votes
                });
        }
    }

}

void Image::DrawDetectedLines(bool parallelVersion)
{
    pixels = originalPixels;
    int maxRho = (int)sqrt(width * width + height * height);   //Dijagonala prostora, max udaljenost od kp

    if (parallelVersion)
    {
        for (const auto& line : detectedLines)
        {
            int rho = line.rho - maxRho;
            int theta = line.theta;

            double thetaRad = theta * numbers::pi / 180.0;

            double a = cos(thetaRad);
            double b = sin(thetaRad);

            //Neka random tacka na pravoj
            double x0 = a * rho;
            double y0 = b * rho;

            int lineLength = max(width, height) * 2;
            //Tacka na 1 kraju
            int x1 = (int)(x0 + lineLength * (-b));
            int y1 = (int)(y0 + lineLength * (a));

            //Tacka na 2. kraju
            int x2 = (int)(x0 - lineLength * (-b));
            int y2 = (int)(y0 - lineLength * (a));
            //Udaljavanje za po lineLength od referentne tacke da se osigura duzina prave preko cele slike

            DrawLine(x1, y1, x2, y2);
        }
    }
    else
    {

        for (const auto& line : serialDetectedLines)
        {
            int rho = line.rho - maxRho;
            int theta = line.theta;

            double thetaRad = theta * numbers::pi / 180.0;

            double a = cos(thetaRad);
            double b = sin(thetaRad);

            //Neka random tacka na pravoj
            double x0 = a * rho;
            double y0 = b * rho;

            int lineLength = max(width, height) * 2;
            //Tacka na 1 kraju
            int x1 = (int)(x0 + lineLength * (-b));
            int y1 = (int)(y0 + lineLength * (a));

            //Tacka na 2. kraju
            int x2 = (int)(x0 - lineLength * (-b));
            int y2 = (int)(y0 - lineLength * (a));
            //Udaljavanje za po lineLength od referentne tacke da se osigura duzina prave preko cele slike

            DrawLine(x1, y1, x2, y2);
        }
    }
}

void Image::DrawLine(int x1, int y1, int x2, int y2)
{
    int steps = max(abs(x2 - x1), abs(y2 - y1));

    for (int i = 0; i <= steps; i++)
    {
        //Trenutne koordinate tacke koja se boji
        int x = x1 + (x2 - x1) * i / steps;
        int y = y1 + (y2 - y1) * i / steps;

        if (x >= 0 && x < width &&               //Proverava da li je pixel unutar slike
            y >= 0 && y < height)
        {
            pixels[y * width + x] = { 255, 0, 0 };
        }
    }
}




bool Image::SaveEdges(string path)
{
    if (edgePixels.empty())
    {
        return false;
    }

    int result = stbi_write_png(
        path.c_str(),
        width,
        height,
        1,                  // 1 kanal (grayscale)
        edgePixels.data(),
        width
    );

    return result != 0;
}

