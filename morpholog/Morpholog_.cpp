#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	Mat upimg = imread("input_r_u.jpg", IMREAD_GRAYSCALE);
	Mat downimg = imread("input_r_d.jpg", IMREAD_GRAYSCALE);
	Mat leftimg = imread("input_l.jpg", IMREAD_GRAYSCALE);

	Mat right_origin_img, origin_img;
	vconcat(upimg, downimg, right_origin_img);
	hconcat(leftimg, right_origin_img, origin_img);

	imshow("origin", origin_img);

	threshold(upimg, upimg, 180, 255, THRESH_BINARY_INV);
	threshold(downimg, downimg, 200, 255, THRESH_BINARY_INV);
	threshold(leftimg, leftimg, 200, 255, THRESH_BINARY_INV);

	Mat skel(upimg.size(), CV_8UC1, Scalar(0));
	Mat skel_1(downimg.size(), CV_8UC1, Scalar(0));
	Mat skel_2(leftimg.size(), CV_8UC1, Scalar(0));

	Mat element_e = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat element_c = getStructuringElement(MORPH_CROSS, Size(3, 3));

	Mat uptemp, uperoded, downtemp, downeroded, lefttemp, lefteroded;

	int i = 0;
	do {
		erode(upimg, uperoded, element_e);
		dilate(uperoded, uptemp, element_e);
		subtract(upimg, uptemp, uptemp);
		bitwise_or(skel, uptemp, skel);
		uperoded.copyTo(upimg);
		i++;
	} while (i != 15);

	i = 0;
	do {
		erode(downimg, downeroded, element_c);
		dilate(downeroded, downtemp, element_c);
		subtract(downimg, downtemp, downtemp);
		bitwise_or(skel_1, downtemp, skel_1);
		downeroded.copyTo(downimg);
		i++;
	} while (i != 25);

	i = 0;
	do {
		erode(leftimg, lefteroded, element_e);
		dilate(lefteroded, lefttemp, element_e);
		subtract(leftimg, lefttemp, lefttemp);
		bitwise_or(skel_2, lefttemp, skel_2);
		lefteroded.copyTo(leftimg);
		i++;
	} while (i != 15);

	Mat right_img, final_img;
	vconcat(skel, skel_1, right_img);
	hconcat(skel_2, right_img, final_img);

	imshow("final", final_img);

	waitKey(0);
	return 0;
}