#include <iostream>
#include <fstream>
#include <sstream>
#include <flann/flann.hpp>

using namespace std;
using namespace flann;

template <typename Index, typename Distance>
float compareWithGroundTruth(Index& index, const Matrix<typename Distance::ElementType>& inputData,
                             const Matrix<typename Distance::ElementType>& testData,
                             const Matrix<size_t>& matches, SearchParams searchParams, const Distance& distance)
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    size_t* indices = new size_t[1];
    DistanceType* dists = new DistanceType[1];

    Matrix<size_t> indices_mat(indices, 1, 1);
    Matrix<DistanceType> dists_mat(dists, 1, 1);

    size_t* neighbors = indices;

    int correct = 0;
    DistanceType distR = 0;
    StartStopTimer t;
    int repeats = 0;

    while (t.value<0.2) {
        repeats++;
        t.start();
        correct = 0;
        distR = 0;
        for (size_t i = 0; i < testData.rows; i++) {
            index.knnSearch(Matrix<ElementType>(testData[i], 1, testData.cols), indices_mat, dists_mat, 1, searchParams);

            correct += countCorrectMatches(neighbors,matches[i], 1);
            distR += computeDistanceRaport<Distance>(inputData, testData[i], neighbors, matches[i], (int)testData.cols, 1, distance);
        }
        t.stop();
    }
    float time = float(t.value/repeats);

    delete[] indices;
    delete[] dists;

    float precicion = (float)correct/(testData.rows);

    DistanceType dist = distR/ testData.rows;

    cout << searchParams.checks << '\t' <<  precicion << '\t' <<  time
         << '\t' <<  1000.0 * time / testData.rows << '\t' <<  dist << endl;

    return precicion;
}

template <typename Index, typename Distance>
void testPrecision(Index &index, const Matrix<typename Distance::ElementType> &inputData,
                    const Matrix<typename Distance::ElementType> &testData, const Matrix<size_t> &matches,
                    const Distance &distance)
{
    cout << "Nodes\tPrecision(%)\tTime(s)\tTime/vec(ms)\tMean dist\n";
    cout << "-----\t------------\t-------\t------------\t---------\n";

    int checks = 1;
    float precision;

    precision = compareWithGroundTruth(index, inputData, testData, matches, SearchParams(checks), distance);
    if (precision > 0.99) {
        return;
    }

    while (precision < 0.85) {
        checks *= 2;
        precision = compareWithGroundTruth(index, inputData, testData, matches, SearchParams(checks), distance);
    }

    while (precision < 0.95) {
        checks += checks / 2;
        precision = compareWithGroundTruth(index, inputData, testData, matches, SearchParams(checks), distance);
    }

    while (precision < 0.99) {
        checks += checks / 4;
        precision = compareWithGroundTruth(index, inputData, testData, matches, SearchParams(checks), distance);
    }
}

template <class T>
void readFile(ifstream &file, vector<T> &video, string filename) {
    T f;

    while(file.read(reinterpret_cast<char *>(&f), sizeof(T))) {
        video.push_back(f);
    }
    cout << "Values length for " << filename << ": " << video.size() << endl;
}

template <class T>
void process_dataset(string filename1, string filename2, unsigned dimensions) {
    ifstream file1(filename1.data(), ios::binary);
    ifstream file2(filename2.data(), ios::binary);

    if (!file1.is_open()) {
        cerr << "Error opening file " << filename1 << endl;
        return;
    }

    if (!file2.is_open()) {
        cerr << "Error opening file " << filename2 << endl;
        return;
    }

    vector<T> video1;
    vector<T> video2;

    readFile(file1, video1, filename1);
    readFile(file2, video2, filename2);

    cout << "Files loaded to memory" << endl;
    Matrix<T> dataset(&video1[0], video1.size() / dimensions, dimensions);
    Matrix<T> query(&video2[0], video2.size() / dimensions, dimensions);

    cout << "Matrix built:\n";
    cout << "Dataset size: " << dataset.rows << endl;
    cout << "Query size: " << query.rows << endl;

    StartStopTimer timer;

    cout << "Building ground through\n";
    timer.start();

    Matrix<size_t> gt(new size_t[query.rows], query.rows, 1);

    compute_ground_truth(dataset, query, gt, 0, L2<T>());
    timer.stop();
    cout << "Ground through built, elapsed time: " << timer.value << endl;

    Index<L2<T> > kdTreeIndex(dataset, KDTreeIndexParams(1));
    Index<L2<T> > randKdTree5Index(dataset, KDTreeIndexParams(5));
    Index<L2<T> > randKdTree20Index(dataset, KDTreeIndexParams(20));
    Index<L2<T> > kMeans5Index(dataset, KMeansIndexParams(5));
    Index<L2<T> > kMeans20Index(dataset, KMeansIndexParams(20));

    cout << "Building indexes" << endl;
    timer.reset();
    timer.start();
    kdTreeIndex.buildIndex();
    randKdTree5Index.buildIndex();
    randKdTree20Index.buildIndex();
    kMeans5Index.buildIndex();
    kMeans20Index.buildIndex();
    timer.stop();
    cout << "Finishing indexation, elapsed time: " << timer.value << endl;

    cout << "KD-Tree\n";
    testPrecision(kdTreeIndex, dataset, query, gt, L2<T>());
    cout << "Random 5 KD-Tree\n";
    testPrecision(randKdTree5Index, dataset, query, gt, L2<T>());
    cout << "Random 20 KD-Tree\n";
    testPrecision(randKdTree20Index, dataset, query, gt, L2<T>());
    cout << "KMeans 5\n";
    testPrecision(kMeans5Index, dataset, query, gt, L2<T>());
    cout << "KMeans 4\n";
    testPrecision(kMeans20Index, dataset, query, gt, L2<T>());
    cout << "Test finished\n";
}

int main(int argc, char *argv[]) {
    string file1, file2, type, dimensionString;
    unsigned dimensions;

    if (argc < 5) {
        cout << "Usage: file1 file2 dimensions type\n";
        cout << "type values:\n";
        cout << "byte (8 bits, integer)\n";
        cout << "float (32 bit, decimal)\n";
    }
    file1 = argv[1];
    file2 = argv[2];
    type = argv[3];
    dimensionString = argv[4];

    stringstream dimensionsStream(dimensionString);
    dimensionsStream >> dimensions;

    if (type == "byte") {
        process_dataset<char>(file1, file2, dimensions);
    } else if (type == "float") {
        process_dataset<float>(file1, file2, dimensions);
    } else {
        cout << "Invalid type\n";
    }
}
