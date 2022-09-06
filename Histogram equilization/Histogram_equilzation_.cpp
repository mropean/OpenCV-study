#include <opencv.hpp>
#include <imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	Mat img, gray_img, gray_result_img, color_img, color_result_img;		// 원본이미지, 흑백이미지, 흑백 결과, 컬러이미지, 컬러 결과
	img = imread("lenna.jpg", IMREAD_COLOR);				// 원본
	cvtColor(img, gray_img, COLOR_BGR2GRAY);				// 원본을 흑백으로 하나 만들기
	color_img = img.clone();							//  원본의 클론 생성

	// 입력영상 흑백 히스토그램 계산
	int Gray_histogram[256] = { 0, };
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			int value = gray_img.at<uchar>(y, x);			// 흑백 영상의 화소값 계산
			Gray_histogram[value]++;
		}
	}

	// (흑백) 히스토그램의 누적 히스토그램 계산
	int Gray_acc_histogram[256] = { 0, };
	int g_acc_sum = 0;
	for (int i = 0; i < 256; i++) {
		g_acc_sum += Gray_histogram[i];
		Gray_acc_histogram[i] = g_acc_sum;
	}

	// (흑백) 히스토그램 평활화 및 결과 영상 히스토그램 계산
	gray_result_img = Mat(img.rows, img.cols, CV_8UC1);
	for (int x = 0; x < gray_img.cols; x++) {
		for (int y = 0; y < gray_img.rows; y++) {
			gray_result_img.at<uchar>(y, x)
				= saturate_cast<uchar>(((float)Gray_acc_histogram[gray_img.at<uchar>(y, x)] / Gray_acc_histogram[255]) * 255);
		}
	}

	// 결과영상 흑백 히스토그램 계산
	int result_Gray_histogram[256] = { 0, };
	for (int x = 0; x < gray_result_img.cols; x++) {
		for (int y = 0; y < gray_result_img.rows; y++) {
			int value = gray_result_img.at<uchar>(y, x);			// 결과 흑백 영상의 화소값 계산
			result_Gray_histogram[value]++;
		}
	}

	// 입력영상 컬러 히스토그램 계산
	int color_histogram[3][256] = { 0, };
	for (int z = 0; z < 3; z++) {
		for (int x = 0; x < img.cols; x++) {
			for (int y = 0; y < img.rows; y++) {
				int value = color_img.at<Vec3b>(y, x)[z];		// 컬러 영상의 화소값 계산
				color_histogram[z][value]++;
			}
		}
	}

	// (컬러) 히스토그램의 누적 히스토그램 계산
	int color_acc_histogram[3][256] = { 0, };
	int c_acc_sum = 0;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 256; i++) {
			c_acc_sum += color_histogram[j][i];
			color_acc_histogram[j][i] = c_acc_sum;
		}
		c_acc_sum = 0;							// 한 채널의 누적값이 끝나면 초기화
	}

	// (컬러) 히스토그램 평활화 및 결과 영상 히스토그램 계산
	color_result_img = Mat(img.rows, img.cols, CV_8UC3);
	for (int z = 0; z < 3; z++) {
		for (int x = 0; x < color_img.cols; x++) {
			for (int y = 0; y < color_img.rows; y++) {
				color_result_img.at<Vec3b>(y, x)[z]
					= saturate_cast<uchar>(
						((float)color_acc_histogram[z][color_img.at<Vec3b>(y, x)[z]] / color_acc_histogram[z][255]) * 255);
			}
		}
	}

	// 결과영상 컬러 히스토그램 계산
	int result_color_histogram[3][256] = { 0, };
	for (int z = 0; z < 3; z++) {
		for (int x = 0; x < color_result_img.cols; x++) {
			for (int y = 0; y < color_result_img.rows; y++) {
				int value = color_result_img.at<Vec3b>(y, x)[z];	// 컬러 영상의 화소값 계산
				result_color_histogram[z][value]++;
			}
		}
	}

	// 히스토그램 그리기
	// (흑백)
	/*{*/
	Mat Draw_gray_histogram = Mat(300, 300, CV_8UC1, Scalar(255));
	Mat Draw_gray_acchistogram = Mat(300, 260, CV_8UC1, Scalar(255));

	// (흑백) 각 히스토그램의 최대값 찾기
	int h_max = -1;
	for (int i = 0; i < 256; i++)
		if (h_max < Gray_histogram[i]) h_max = Gray_histogram[i];

	int acc_h_nax = -1;
	for (int i = 0; i < 256; i++)
		if (acc_h_nax < result_Gray_histogram[i]) acc_h_nax = result_Gray_histogram[i];

	for (int i = 0; i < 256; i++) {
		int hist = 300 * Gray_histogram[i] / (float)h_max;
		line(Draw_gray_histogram, Point(i + 2, 300), Point(i + 2, 300 - hist), Scalar(128, 128, 128));

		int hist2 = 300 * result_Gray_histogram[i] / (float)acc_h_nax;
		line(Draw_gray_acchistogram, Point(i + 2, 300), Point(i + 2, 300 - hist2), Scalar(128, 128, 128));
	}
	/*}*/

	// (컬러)
	/*{*/
	Mat Draw_color_histogram = Mat(300, 300, CV_8UC3, Scalar(255,255,255));
	Mat Draw_color_acchistogram = Mat(300, 260, CV_8UC3, Scalar(255,255,255));

	// (컬러) 각 히스토그램의 최대값 찾기 (노가다방법)
	// 기본 히스토그램
	int blue_max = -1;
	int green_max = -1;
	int red_max = -1;
	for (int i = 0; i < 256; i++)
		if (blue_max < color_histogram[0][i]) blue_max = color_histogram[0][i];
	for (int i = 0; i < 256; i++)
		if (green_max < color_histogram[1][i]) green_max = color_histogram[1][i];
	for (int i = 0; i < 256; i++)
		if (red_max < color_histogram[2][i]) red_max = color_histogram[2][i];

	// 누적 히스토그램
	int acc_blue_max = -1;
	int acc_green_max = -1;
	int acc_red_max = -1;
	for (int i = 0; i < 256; i++)
		if (acc_blue_max < result_color_histogram[0][i]) acc_blue_max = result_color_histogram[0][i];
	for (int i = 0; i < 256; i++)
		if (acc_green_max < result_color_histogram[1][i]) acc_green_max = result_color_histogram[1][i];
	for (int i = 0; i < 256; i++)
		if (acc_red_max < result_color_histogram[2][i]) acc_red_max = result_color_histogram[2][i];

	// 그리기
	for (int i = 0; i < 256; i++) {
		int color_hist_b = 300 * color_histogram[0][i] / (float)blue_max;
		line(Draw_color_histogram, Point(i + 2, 300), Point(i + 2, 300 - color_hist_b), Scalar(255, 0, 0));
		int color_hist_g = 300 * color_histogram[1][i] / (float)green_max;
		line(Draw_color_histogram, Point(i + 2, 300), Point(i + 2, 300 - color_hist_g), Scalar(0, 255, 0));
		int color_hist_r = 300 * color_histogram[2][i] / (float)red_max;
		line(Draw_color_histogram, Point(i + 2, 300), Point(i + 2, 300 - color_hist_r), Scalar(0, 0, 255));

		int color_hist2_b = 300 * result_color_histogram[0][i] / (float)acc_blue_max;
		line(Draw_color_acchistogram, Point(i + 2, 260), Point(i + 2, 300 - color_hist2_b), Scalar(255, 0, 0));
		int color_hist2_g = 300 * result_color_histogram[1][i] / (float)acc_green_max;
		line(Draw_color_acchistogram, Point(i + 2, 300), Point(i + 2, 300 - color_hist2_g), Scalar(0, 255, 0));
		int color_hist2_r = 300 * result_color_histogram[2][i] / (float)acc_red_max;
		line(Draw_color_acchistogram, Point(i + 2, 300), Point(i + 2, 300 - color_hist2_r), Scalar(0, 0, 255));
	}

	/*}*/

	Mat Drawing_gray_histogram_result, Drawing_color_histogram_result;
	hconcat(Draw_gray_histogram, Draw_gray_acchistogram, Drawing_gray_histogram_result);
	hconcat(Draw_color_histogram, Draw_color_acchistogram, Drawing_color_histogram_result);

	imshow("원본 영상", img);
	imshow("흑백결과 영상", gray_result_img);
	imshow("흑백 영상의 히스토그램", Drawing_gray_histogram_result);
	imshow("컬러 결과 영상", color_result_img);
	imshow("컬러 영상의 히스토그램", Drawing_color_histogram_result);

	waitKey(0);
	return 0;
}