#include <iostream>
#include <fstream>
#include <sstream>
#include <flann/flann.hpp>

using namespace std;

template<class T>
void testAlgorithm(const flann::Matrix<T> &dataset, const flann::Matrix<T> &query, const flann::Matrix<size_t> &gt,
                   flann::Index<flann::L2<T> > &index) {
    int checks;
    float time;

    cout << "Testing KDTree\n";
    time = test_index_precision(index, dataset, query, gt, 0.6, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.7, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.8, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.85, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.9, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.95, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 0.99, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
    time = test_index_precision(index, dataset, query, gt, 1.0, checks, flann::L2<T>());
    cout << "Elapsed time: " << time << endl;
}

template <class T>
void process_dataset(string filename1, string filename2, unsigned dimensions) {
    ifstream file1(filename1, ios::binary);
    ifstream file2(filename2, ios::binary);

    if (!file1.is_open()) {
        cerr << "Error opening file " << filename1 << endl;
        return;
    }

    if (!file2.is_open()) {
        cerr << "Error opening file " << filename2 << endl;
        return;
    }

    vector<T> video1((istreambuf_iterator<T>(file1)), istreambuf_iterator<T>());
    vector<T> video2((istreambuf_iterator<T>(file2)), istreambuf_iterator<T>());

    cout << "Files loaded to memory" << endl;
    flann::Matrix<T> dataset(&video1[0], video1.size() / dimensions, dimensions);
    flann::Matrix<T> query(&video2[0], video2.size() / dimensions, dimensions);

    flann::StartStopTimer timer;

    flann::Matrix<size_t> gt(new size_t[query.rows], query.rows, 1);
    cout << "Building grand through\n";
    timer.start();
    flann::compute_ground_truth(dataset, query, gt, 0, flann::L2<T>());
    timer.stop();
    cout << "GrandThrough built, elapsed time: " << timer.value << endl;

    flann::Index<flann::L2<T> > kdTreeIndex(dataset, flann::KDTreeSingleIndexParams());
    flann::Index<flann::L2<T> > randKdTree5Index(dataset, flann::KDTreeIndexParams(5));
    flann::Index<flann::L2<T> > randKdTree20Index(dataset, flann::KDTreeIndexParams(20));
    flann::Index<flann::L2<T> > kMeans5Index(dataset, flann::KMeansIndexParams(5));
    flann::Index<flann::L2<T> > kMeans20Index(dataset, flann::KMeansIndexParams(20));

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

    testAlgorithm(dataset, query, gt, kdTreeIndex);
    testAlgorithm(dataset, query, gt, randKdTree5Index);
    testAlgorithm(dataset, query, gt, randKdTree20Index);
    testAlgorithm(dataset, query, gt, kMeans5Index);
    testAlgorithm(dataset, query, gt, kMeans20Index);
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
    }
}