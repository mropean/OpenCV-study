#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	Mat img = imread("lenna.jpg", IMREAD_COLOR);
	Mat rimg;
	rimg = img.clone();

	// 3x3 = 9, 5x5 = 25, ...
	int wsize = 5;	// 가로
	int hsize = 5;	// 세로
	float sigma = 1.0;	// 표준편차
	float bsigma = 10.0;	
	float sum = 0;	// 합
	float inputsum = 0;
	int run = 20;
	int run_count = 0;
	float* distance = new float[(wsize * hsize)];
	float* Gmask = new float[(wsize * hsize)];
	float* Bmask = new float[(wsize * hsize)];

	// 마스크 요소 구하기
	for (int j = 0; j < hsize; j++)
		for (int i = 0; i < wsize; i++) {
			distance[j * wsize + i] = sqrt(pow(i - wsize / 2, 2) + pow(j - hsize / 2, 2));
		}

	// 가우시안 마스크 구하기
	for (int i = 0; i < wsize * hsize; i++) {
		Gmask[i] = exp(-pow(distance[i], 2) / (2 * pow(sigma,2)));
		sum += Gmask[i];
	}

	// 마스크합 1로 만들기
	for (int i = 0; i < wsize * hsize; i++) {
		Gmask[i] /= sum;
	}

A:
	// input 이미지의 각 픽셀값 구하기
	for (int z = 0; z < 3; z++) {
		for (int y = hsize / 2; y < img.rows - hsize / 2; y++) {
			for (int x = wsize / 2; x < img.cols - wsize / 2; x++) {
				for (int j = 0; j < hsize; j++) {
					for (int i = 0; i < wsize; i++) {
						if (run == 1) {	Bmask[j * wsize + i] = img.at<Vec3b>(j + y - hsize / 2, i + x - wsize / 2)[z];}
						else{ Bmask[j * wsize + i] = rimg.at<Vec3b>(j + y - hsize / 2, i + x - wsize / 2)[z]; }
					}
				}

				float middle = Bmask[(wsize * hsize) / 2];

				sum = 0;
				for (int a = 0; a < wsize * hsize; a++) {
						Bmask[a] = Gmask[a] * exp(-pow(Bmask[a] - middle, 2) / (2 * pow(bsigma, 2)));
						sum += Bmask[a];
				}

				for (int a = 0; a < wsize * hsize; a++) {
						Bmask[a] /= sum;
				}

				inputsum = 0;
				// 곱하기
				for (int j = 0; j < hsize; j++) {
					for (int i = 0; i < wsize; i++) {
						if (run == 1) { inputsum += (Bmask[j * wsize + i] * img.at<Vec3b>(j + y - hsize / 2, i + x - wsize / 2)[z]); }
						else { inputsum += (Bmask[j * wsize + i] * rimg.at<Vec3b>(j + y - hsize / 2, i + x - wsize / 2)[z]); }
					}
				}
				rimg.at<Vec3b>(y, x)[z] = saturate_cast<uchar>(inputsum);
			}
		}
	}
	run_count++;
	if (run != run_count) { goto A;}

	imshow("img", img);
	imshow("bfimg", rimg);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
							// x, y
	Sobel(img, grad_x, CV_8U, 1, 0);
	Sobel(img, grad_y, CV_8U, 0, 1);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	Mat grad;
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	
	Mat quntam;
	threshold(grad, quntam, 50, 255, THRESH_BINARY_INV);

	Mat cartoon_gradi, cartoon_quantom;
	addWeighted(rimg, 0.45, ~grad, 0.60, 0, cartoon_gradi);
	addWeighted(rimg, 0.75, quntam, 0.25, 0, cartoon_quantom);
	imshow("cartoon_gradiant", cartoon_gradi);
	imshow("cartoon_quantom", cartoon_quantom);

	waitKey(0);

	delete distance, Gmask, Bmask;
	return 0;
}