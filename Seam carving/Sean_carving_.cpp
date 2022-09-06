#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int distTo[2000][2000];
int edgeTo[2000][2000];
int width;
int height;

Mat src = imread("Broadway.png", IMREAD_COLOR);
Mat lab_vertical;
Mat lab_horizon;
Mat copy_src;

// sobel edge 검출
Mat computeFullEnergy(Mat img) {
	Mat energy(img.rows, img.cols, CV_8UC1, Scalar(255));
	int i, j, gx, gy;
	for (i = 1; i < img.rows - 1; i++) {
		for (j = 1; j < img.cols - 1; j++) {
			gx = gy = 0;
			gx = abs((img.at<Vec3b>(i - 1, j + 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i, j + 1)[0] - img.at<Vec3b>(i, j - 1)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i + 1, j - 1)[0]));

			gy = abs((img.at<Vec3b>(i + 1, j - 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i + 1, j)[0] - img.at<Vec3b>(i - 1, j)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i - 1, j + 1)[0]));
			energy.at<uchar>(i, j) = abs(gx + gy);
		}
	}

	return energy;
}

vector<uint> findVerticalSeam(Mat img) {
	Mat energy = computeFullEnergy(img);
	vector<uint> seam(img.rows);		// unsinged int 형식의 인풋 이미지의 rows 크기의 벡터생성

	// 엣지 배열, 거리 배열 초기화
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// 모든 외곽들은 0으로 초기화 후 외곽을 제외한 안쪽은 int형의 최대값으로 설정
			distTo[i][j] = (i == 0) ? 0 : numeric_limits<int>::max();
			edgeTo[i][j] = 0;
		}
	}

	// 가중치 구하기
	for (int r = 0; r < img.rows - 1; r++) {
		for (int c = 0; c < img.cols; c++) {
			// 왼쪽 엣지 제외
			if (c != 0) {
				//  가운데 기준 아래의 왼쪽
				if (distTo[r + 1][c - 1] > (distTo[r][c] + (int)energy.at<uchar>(r + 1, c - 1))) {
					distTo[r + 1][c - 1] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c - 1);
					edgeTo[r + 1][c - 1] = 1;
				}
			}

			// 가운데 기준 바로 아래
			if (distTo[r + 1][c] > (distTo[r][c] + (int)energy.at<uchar>(r + 1, c))) {
				distTo[r + 1][c] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c);
				edgeTo[r + 1][c] = 0;
			}

			// 맨 오른쪽 외곽 제외
			if (c != (img.cols - 1)) {
				// 가운데 기준 아래의 오른쪽
				if (distTo[r + 1][c + 1] > (distTo[r][c] + energy.at<uchar>(r + 1, c + 1))) {
					distTo[r + 1][c + 1] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c + 1);
					edgeTo[r + 1][c + 1] = -1;
				}
			}
		}
	}

	// 맨 아래 row에서 최소값 가진 col인덱스 찾기
	int min_index = 0, min = distTo[img.rows - 1][0];
	for (int i = 1; i < img.cols; i++) {
		if (distTo[img.rows - 1][i] < min) {
			min = distTo[img.rows - 1][i];
			min_index = i;
		}
	}

	// 위에서 찾은 인덱스가 심 벡터의 원소로 뒤에서 부터 삽입
	seam[img.rows - 1] = min_index;

	// 아래에서 부터 위로 올라가며 그 다음 row의 제일 최솟값 가진 col 인덱스 찾기
	for (int i = img.rows - 1; i > 0; --i) {
		// 심 벡터의 위 인덱스의 이전 인덱스가 된다.
		seam[i - 1] = seam[i] + edgeTo[i][seam[i]];
	}

	// 구해진 심 반환
	return seam;
}

Mat removeVerticalSeam(Mat img, vector<uint> seam) {
	// 구해진 심 인덱스로 각 row에서 최솟값인 col를 제거한다.
	for (int r = 0; r < img.rows; r++) {
		for (int c = seam[r]; c < img.cols - 1; c++) {
			img.at<Vec3b>(r, c) = img.at<Vec3b>(r, c + 1);
		}
	}

	// 이후 다시 영상을 재구성한다.
	img = img(Rect(0, 0, img.cols - 1, img.rows));
	return img;
}

vector<uint> findHorizonSeam(Mat img) {
	Mat energy = computeFullEnergy(img);
	vector<uint> seam(img.cols);		// unsinged int 형식의 인풋 이미지의 cols 크기의 벡터생성

	// 엣지 배열, 거리 배열 초기화
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			// 모든 외곽들은 0으로 초기화 후 외곽을 제외한 안쪽은 int형의 최대값으로 설정
			distTo[i][j] = (i == 0) ? 0 : numeric_limits<int>::max();
			edgeTo[i][j] = 0;
		}
	}

	// 가중치 구하기
	for (int c = 0; c < img.cols - 1; c++) {
		for (int r = 0; r < img.rows; r++) {
			if (r != 0) { // 위쪽 엣지 제외 
				//  가운데 기준 오른쪽 위
				if (distTo[r - 1][c + 1] > (distTo[r][c] + (int)energy.at<uchar>(r - 1, c + 1))) {
					distTo[r - 1][c + 1] = distTo[r][c] + (int)energy.at<uchar>(r - 1, c + 1);
					edgeTo[r - 1][c + 1] = 1;
				}
			}
			// 가운데 기준 바로 오른쪽
			if (distTo[r][c + 1] > (distTo[r][c] + (int)energy.at<uchar>(r, c + 1))) {
				distTo[r][c + 1] = distTo[r][c] + (int)energy.at<uchar>(r, c + 1);
				edgeTo[r][c + 1] = 0;
			}
			// 맨 아래쪽 외곽 제외
			if (r != (img.rows - 1)) {
				// 가운데 기준 오른쪽 아래
				if (distTo[r + 1][c + 1] > (distTo[r][c] + energy.at<uchar>(r + 1, c + 1))) {
					distTo[r + 1][c + 1] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c + 1);
					edgeTo[r + 1][c + 1] = -1;
				}
			}
		}
	}

	// 맨 오른쪽 col에서 최소값 가진 row인덱스 찾기
	int min_index = 0, min = distTo[img.cols - 1][0];
	for (int i = 1; i < img.rows; i++) {
		if (distTo[i][img.cols - 1] < min) {
			min = distTo[i][img.cols - 1];
			min_index = i;
		}
	}

	// 위에서 찾은 인덱스가 심 벡터의 원소로 뒤에서 부터 삽입
	seam[img.cols - 1] = min_index;

	// 아래에서 부터 위로 올라가며 그 다음 row의 제일 최솟값 가진 col 인덱스 찾기
	for (int i = img.cols - 1; i > 0; --i) {
		// 심 벡터의 위 인덱스의 이전 인덱스가 된다.
		seam[i - 1] = seam[i] + edgeTo[seam[i]][i];
	}

	// 구해진 심 반환
	return seam;
}

Mat removeHorizonSeam(Mat img, vector<uint> seam) {
	// 구해진 심 인덱스로 각 col에서 최솟값인 row를 제거한다.
	for (int c = 0; c < img.cols; c++) {
		for (int r = seam[c]; r < img.rows - 1; r++) {
			img.at<Vec3b>(r, c) = img.at<Vec3b>(r + 1, c);
		}
	}

	// 이후 다시 영상을 재구성한다.
	img = img(Rect(0, 0, img.cols, img.rows - 1));
	return img;
}

void vertical_trackbar(int, void*) {
	cvtColor(src, lab_vertical, COLOR_RGB2Lab);

	for (int i = 0; i < abs(src.cols - width); i++) {
		vector<uint> seam_Vertical = findVerticalSeam(lab_vertical);	// 심 구하기
		lab_vertical = removeVerticalSeam(lab_vertical, seam_Vertical);	// 구한 심 제거하기
	}
	// sobel 에서 컬러로 변환
	cvtColor(lab_vertical, lab_vertical, COLOR_Lab2RGB);
	imshow("only delete Vertical seam", lab_vertical);
}

void horizon_trackbar(int, void*) {
	cvtColor(src, lab_horizon, COLOR_RGB2Lab);

	for (int i = 0; i < abs(src.rows - height); i++) {
		vector<uint> seam_Horizon = findHorizonSeam(lab_horizon);	// 심 구하기
		lab_horizon = removeHorizonSeam(lab_horizon, seam_Horizon);	// 구한 심 제거하기
	}
	// sobel 에서 컬러로 변환
	cvtColor(lab_horizon, lab_horizon, COLOR_Lab2RGB);
	imshow("only delete Horizon seam", lab_horizon);
}

int main() {
	imshow("only delete Vertical seam", src);
	imshow("only delete Horizon seam", src);

	copy_src = src.clone();
	width = src.cols;
	height = src.rows;

	// 각각의 트랙바 생성
	createTrackbar("Vertical", "only delete Vertical seam", &width, (int)src.cols, vertical_trackbar, &copy_src);
	createTrackbar("Horizon", "only delete Horizon seam", &height, (int)src.rows, horizon_trackbar, &copy_src);

	waitKey(0);
	destroyAllWindows();

	return 0;
}