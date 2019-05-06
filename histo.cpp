/*
 svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 5000, 1e-6));
 
 svm.C = 0.5
 svm.Coef0 = 0
 svm.Degree = 0
 svm.gamma = 1e-05
 svm.Nu = 0
 svm.P = 0
 人工物画像の正解枚数1083/1200
 自然物画像の正解枚数1088/1200
 人工物画像の正解率＝90.25%
 自然物画像の正解率＝90.6667%
 全体の正解率＝90.4583%

SVMの学習時間:243180sec.
svm.C = 0.5
svm.Coef0 = 0
svm.Degree = 0
svm.gamma = 1e-05
svm.Nu = 0
svm.P = 0
人工物画像の正解枚数1088/1200
自然物画像の正解枚数1091/1200
人工物画像の正解率＝90.6667%
自然物画像の正解率＝90.9167%
全体の正解率＝90.7917%

 */

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <math.h>
#include <time.h>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

//近傍画素と参照画素との距離
#define R 2

//人工物と自然物それぞれの教師データの数
#define SIZE  15000

//人工物と自然物それぞれのテストデータの数
#define SIZE2 1200


//画像入手関数
inline void load_images( const String & dirname, vector< Mat > & img_lst)
{
    
    vector< String > files;
    glob( dirname, files );
    
    for (size_t i = 0; i < files.size(); ++i){
        Mat img = imread( files[i] );
        if (img.empty()){
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        cvtColor(img, img, COLOR_BGR2GRAY);
        img_lst.push_back( img );
    }
    
}


template < typename T > string to_string( const T& n )
{
    
    ostringstream stm;
    stm << n;
    return stm.str();
    
}


//共起ヒストグラム関数
inline void kyouki( int height, int width, int m1, int n1, int m2, int n2, Mat h, Mat strength, Mat direction)
{
    
    int i = direction.at<float>(m1,n1) / 180 * 64;
    int j = direction.at<float>(m2,n2) / 180 * 64;
    
    h.at<float>(i,j) += 1.5 * (strength.at<float>(m1,n1) * (strength.at<float>(m2,n2)));
    h.at<float>(i,j) += 0.5 * (strength.at<float>(m1,n1) * (strength.at<float>(m2,n2)));
    
}


//フーリエ変換関数
inline void img_dft(Mat h, vector< Mat > & supectol)
{
    
    Mat furimg;
    
    //実部のみのimageと虚部を0で初期化したMatをRealImaginary配列に入れる
    Mat RealIamginary[] = { h, Mat::zeros(h.size(), CV_32F) };
    
    //配列を合成
    merge(RealIamginary, 2, furimg);
    
    //フーリエ変換
    dft(furimg, furimg);
    
    //furimg.at<float>(0,0) = 0;
    cout << furimg.at<float>(0,0) << endl;
    
    Mat divdisplay[2];
    
    //フーリエ後を実部と虚部に分ける
    split(furimg, divdisplay);
    
    
    Mat display;
    magnitude(divdisplay[0], divdisplay[1], display);
    display = abs(display);
    
    const int halfW = display.cols / 2;
    const int halfH = display.rows / 2;
    
    Mat tmp;
    
    Mat q0(display, Rect(0, 0, halfW, halfH));
    Mat q1(display, Rect(halfW, 0, halfW, halfH));
    Mat q2(display, Rect(0, halfH, halfW, halfH));
    Mat q3(display, Rect(halfW, halfH, halfW, halfH));
    
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    //表示用に正規化
    Mat outdisplay;
    normalize(display, outdisplay, 0, 255, NORM_MINMAX);
    supectol.push_back(outdisplay);
    
    
}


//SVM判定用に画像変換 (64*64->1*4096)
inline void convert_to_ml(const vector< Mat > & train_samples, Mat& trainData)
{
    
    const int rows = (int)train_samples.size();
    const int cols = (int)(train_samples[0].cols * train_samples[0].rows);
    
    trainData = Mat(rows, cols, CV_32F);
    
    for (int i = 0; i < rows; i++){
        for (int m = 0; m < 64; m++){
            for (int n = 0; n < 64; n++){
                trainData.at<float>(i, m * 64 + n) = train_samples[i].at<float>(m,n);
            }
        }
    }
    
}

//スペクトル強度
inline void supectol(vector< Mat > & img, vector< Mat > & supectol, string name1, string name2, string name3)
{
    
    Mat gray_sobelx, gray_sobely;
    Mat edge_strength, edge_direction;
    
    for (unsigned int x = 0; x < img.size(); x++){
        
        cout << x+1 << "/" << img.size() << " ";
        
        
        //フィルタ
        Sobel(img[x], gray_sobelx, CV_32F, 1, 0);
        Sobel(img[x], gray_sobely, CV_32F, 0, 1);
        
        
        //エッジ強度とエッジ方向を求める
        cartToPolar(gray_sobelx, gray_sobely, edge_strength, edge_direction, true);
        if (x == 0){
            imwrite(name3 + ".jpg", edge_strength);
        }
        
        for (int m = 0; m < img[x].rows; m++){
            for (int n = 0; n < img[x].cols; n++){
                if (edge_direction.at<float>(m,n) >= 180){
                    edge_direction.at<float>(m,n) -= 180;
                }
            }
        }
        
        
        //共起ヒストグラム求める
        Mat h = Mat::zeros(Size(64, 64), CV_32F);
        for (int m = R; m < img[x].rows - R; m++){
            for (int n = R; n < img[x].cols - R; n++){
                for (int m_r = (m - R); m_r < (m - R) + (2 * R + 1); m_r++){
                    for (int n_r = (n - R); n_r < (n - R) + (2 * R + 1); n_r++){
                        if ((abs(m - m_r) == R) || (abs(n - n_r) == R)){
                            kyouki(img[x].rows, img[x].cols, m, n, m_r, n_r, h, edge_strength, edge_direction);
                        }
                    }
                }
            }
        }
        
        
        //最大値を255,最小値を0に正規化
        normalize(h, h, 0, 255, NORM_MINMAX);
        if (x == 0){
            imwrite(name1 + ".jpg", h);
        }
        
        
        //平均値を引く
        h = h - mean(h)[0];
        
        
        //スペクトル強度取得
        img_dft(h, supectol);
        if (x == 0){
            imwrite(name2 + ".jpg", supectol[0]);
        }
    }
    
    
}



inline void shuffle(int array[], int size)
{
    
    int i = size;
    srand((unsigned) time(NULL));
    
    while (i > 1) {
        int j = rand() % i;
        i--;
        int t = array[i];
        array[i] = array[j];
        array[j] = t;
    }
    
}



int main(int argc, const char* argv[])
{

    Mat labels;

  
    //人工物画像とラベル(1)の入手
    vector<Mat> img_test_arti;
    vector<Mat> img_test_arti_2;
  
    clog << "人工物画像の訓練データを読み込み中…" << endl;
    load_images("../data/SVM_data/jinkou", img_test_arti);
  
    if ( img_test_arti.size() > 0 ){
        clog << "人工物画像の訓練データ読み込みました" << endl;
    }else{
        clog << "画像がありません " <<endl;
        return 1;
    }


    int value[img_test_arti.size()];
    for (size_t i = 0; i < img_test_arti.size(); i++){
        value[i] = i;
    }


    int size = sizeof(value) / sizeof(int);
    shuffle(value, size);


    for (size_t i = 0; i < SIZE; i++){
        Mat img = img_test_arti[value[i]];
        img_test_arti_2.push_back( img );
        labels.push_back(1);
    }

  
    //自然物画像とラベル(0)の入手
    vector<Mat> img_test_natu;
    vector<Mat> img_test_natu_2;
    
    clog << "自然物画像の訓練データを読み込み中…" << endl;
    load_images("../data/SVM_data/sizen", img_test_natu);
  
    if ( img_test_natu.size() > 0 ){
        clog << "自然物画像の訓練データを読み込みました" << endl;
    }else{
        clog << "画像がありません " <<endl;
        return 1;
    }

 
    int value2[img_test_natu.size()];
    for (size_t i = 0; i < img_test_natu.size(); i++){
        value2[i] = i;
    }

    size = sizeof(value2) / sizeof(int);
    shuffle(value2, size);

    for (size_t i = 0; i < SIZE; i++){
        Mat img = img_test_natu[value2[i]];
        img_test_natu_2.push_back( img );
        labels.push_back(0);
    }

    cout << "人工物画像の読み込み枚数：" << img_test_arti.size() << endl;
    cout << "自然物画像の読み込み枚数：" << img_test_natu.size() << endl;
    cout << "人工物画像の学習枚数：" << img_test_arti_2.size() << endl;
    cout << "自然物画像の学習枚数：" << img_test_natu_2.size() << endl;
    cout << "学習データの合計枚数：" << img_test_arti_2.size() + img_test_natu_2.size() << endl;

    vector<Mat> img_test;

    vector<Mat> supectol_test_arti;
    vector<Mat> supectol_test_natu;
    vector<Mat> supectol_test;

    clock_t start;
    clock_t end;


    string name1 = "arti_histo_0";
    string name2 = "arti_supectol_0";
    string name3 = "sizen_histo_0";
    string name4 = "sizen_supectol_0";
    string name5 = "arti_edge_0";
    string name6 = "sizen_ege_0";
  
    //スペクトルの入手
    cout << "人工物画像の訓練データのスペクトル強度所得中..." << endl;
    imwrite("arti_ori_0.jpg", img_test_arti_2[0]);
    supectol(img_test_arti_2, supectol_test_arti, name1, name2, name5);
    cout << "人工物画像の訓練データのスペクトル強度を所得しました" << endl;

    cout << "自然物画像の訓練データのスペクトル強度所得中..." << endl;
    imwrite("sizen_ori_0.jpg", img_test_natu_2[0]);
    supectol(img_test_natu_2, supectol_test_natu, name3, name4, name6);
    cout << "自然物画像の訓練データのスペクトル強度を所得しました" << endl;

    for (size_t i = 0; i < img_test_arti_2.size(); i++){
        supectol_test.push_back(supectol_test_arti[i]);
    }

    for (size_t i = 0; i < img_test_natu_2.size(); i++){
        supectol_test.push_back(supectol_test_natu[i]);
    }
 

    Mat train_data;
    convert_to_ml(supectol_test, train_data);
    imwrite("train_data.jpg",train_data);


    //SVMの構築
    String svm_name = "svm_model_ex.xml";

    start = clock();

    cout << "SVM学習中..." << endl;

    static Ptr< SVM > svm = SVM::create();

    svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 5000, 1e-6));
    Ptr<TrainData> trainDataPtr = TrainData::create(train_data, ROW_SAMPLE, labels);
    svm->trainAuto(trainDataPtr);
  
    // 終了時間
    cout << "SVMの学習が終了しました" << endl;
    end = clock();
    cout << "SVMの学習時間:" << (double)(end - start) / CLOCKS_PER_SEC << "sec." << endl;

    svm->save(svm_name);



    Mat descriptor = Mat::zeros(Size(64 * 64, 1), CV_32F);
    int decision;
 
    vector<Mat> img_eva_jinkou;
    vector<Mat> img_eva_sizen;
    vector<Mat> img_eva_jinkou_2;
    vector<Mat> img_eva_sizen_2;
    vector<Mat> supectol_eva_jinkou;
    vector<Mat> supectol_eva_sizen;
    vector<Mat> supectol_eva_tmp;

    Mat test_jinkou;
    Mat test_sizen;

    float count_a = 0;
    float count_n = 0;

    float kekka_a;
    float kekka_n;
    float kekka_z;


  
    //判別用人工物画像の入手
    clog << "人工物画像のテストデータを読み込み中…" << endl;
    load_images("../data/CNN_data/test/jinkou", img_eva_jinkou);
    
    if (img_eva_jinkou.size() > 0){
        clog << "人工物画像のテストデータを読み込みました" << endl;
    }else{
        clog << "画像がありません " <<endl;
        return 1;
    }
    
    
    int value3[img_eva_jinkou.size()];
    for (size_t i = 0; i <  img_eva_jinkou.size(); i++){
        value3[i] = i;
    }
  
    size = sizeof(value3) / sizeof(int);
    shuffle(value3, size);

    for (size_t i = 0; i < SIZE2; i++){
        Mat img =  img_eva_jinkou[value3[i]];
        img_eva_jinkou_2.push_back( img );
    }


    //判別用自然物画像の入手
    clog << "自然物画像のテストデータを読み込み中…" << endl;
    load_images("../data/CNN_data/test/sizen", img_eva_sizen);
  
    if (img_eva_sizen.size() > 0){
        clog << "自然物画像のテストデータを読み込みました" << endl;
    }else{
        clog << "画像がありません " <<endl;
        return 1;
    }
    
    
    int value4[img_eva_sizen.size()];
    for (size_t i = 0; i <  img_eva_sizen.size(); i++){
        value4[i] = i;
    }
  
    size = sizeof(value4) / sizeof(int);
    shuffle(value4, size);

    for (size_t i = 0; i < SIZE2; i++){
        Mat img =  img_eva_sizen[value4[i]];
        img_eva_sizen_2.push_back( img );
    }


    cout << "人工物画像の読み込み枚数：" << img_eva_jinkou.size() << endl;
    cout << "自然物画像の読み込み枚数：" << img_eva_sizen.size() << endl;
    cout << "人工物画像のテスト枚数：" << img_eva_jinkou_2.size() << endl;
    cout << "自然物画像のテスト枚数：" << img_eva_sizen_2.size() << endl;
    cout << "テストデータの合計枚数：" << img_eva_jinkou_2.size() + img_eva_sizen_2.size() << endl;

    
    string name7 = "arti_histo_1";
    string name8 = "arti_supectol_1";
    string name9 = "sizen_histo_1";
    string name10 = "sizen_supectol_1";
    string name11 = "arti_edge_1";
    string name12 = "sizen_ege_1";
  
    //スペクトルの入手
    cout << "人工物画像のテストデータのスペクトル強度所得中..." << endl;
    imwrite("arti_ori_1.jpg", img_eva_jinkou_2[0]);
    supectol(img_eva_jinkou_2, supectol_eva_jinkou, name7, name8, name11);
    convert_to_ml(supectol_eva_jinkou, test_jinkou);
    cout << "人工物画像のテストデータのスペクトル強度を所得しました" << endl;

    cout << "自然物画像のテストデータのスペクトル強度所得中..." << endl;
    imwrite("sizen_ori_1.jpg", img_eva_sizen_2[0]);
    supectol(img_eva_sizen_2, supectol_eva_sizen, name9, name10, name12);
    convert_to_ml(supectol_eva_sizen, test_sizen);
    cout << "自然物画像のテストデータのスペクトル強度を所得しました" << endl;



    //人工物画像の判別
    cout << "SVMで人工物識別中..." << endl;
    for (int i = 0; i < test_jinkou.rows; i++){
        for (int m = 0; m < test_jinkou.cols; m++){
            descriptor.at<float>(0, m) = test_jinkou.at<float>(i, m);
        }

        decision = (int)svm->predict(descriptor);

        if (decision == 1){
            clog << i << ":人工物画像->人工物画像" << endl;
            imwrite("./result/" + to_string(i) + "_jinkou.jpg", img_eva_jinkou_2[i]);
            count_a++;
        }else{
            clog << i << ":人工物画像->自然物画像" << endl;
            imwrite("./result/" + to_string(i) + "_sizen.jpg", img_eva_jinkou_2[i]);
        }
    }

  

    //自然物画像の判別
    cout << "SVMで自然物識別中..." << endl;
    for (int i = 0; i < test_sizen.rows; i++){
        for (int m = 0; m < test_sizen.cols; m++){
            descriptor.at<float>(0, m) = test_sizen.at<float>(i, m);
        }
    
        decision = (int)svm->predict(descriptor);

        if (decision == 1){
            clog << i << ":自然物画像->人工物画像" << endl;
            imwrite("./result_en/" + to_string(i+50) + "_jinkou.jpg", img_eva_sizen_2[i]);
        }else{
            clog << i << ":自然物画像->自然物画像" << endl;
            imwrite("./result_en/" + to_string(i+50) + "_sizen.jpg", img_eva_sizen_2[i]);
            count_n++;
        }
    }

  

    //識別結果
    kekka_a = count_a / SIZE2;
    kekka_n = count_n / SIZE2;
    kekka_z = (count_a + count_n) / (SIZE2 * 2);

    cout << "SVMの学習時間:" << (double)(end - start) / CLOCKS_PER_SEC << "sec." << endl;
    clog << "svm.C = " << svm->getC() << endl;
    clog << "svm.Coef0 = " << svm->getCoef0() << endl;
    clog << "svm.Degree = " << svm->getDegree() << endl;
    clog << "svm.gamma = " << svm->getGamma() << endl;
    clog << "svm.Nu = " << svm->getNu() << endl;
    clog << "svm.P = " << svm->getP() << endl;
    cout << "人工物画像の正解枚数" << count_a << "/" << img_eva_jinkou_2.size() << endl;
    cout << "自然物画像の正解枚数" << count_n << "/" << img_eva_sizen_2.size() << endl;
    cout << "人工物画像の正解率＝" << kekka_a * 100 << "%" << endl;
    cout << "自然物画像の正解率＝" << kekka_n * 100 << "%" << endl;
    cout << "全体の正解率＝" << kekka_z * 100 << "%" << endl;

    return 0;
  
}

