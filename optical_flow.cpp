#include "opencv2\videoio\videoio.hpp"
#include "opencv2\opencv.hpp"
#include <cstdio>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <math.h>

using namespace cv;
using namespace std;

#define stored_pi  acos(-1.0)				//3.14159265358979323846

inline static double pow_2(int ip_arg)
{
	return ip_arg * ip_arg;
}

inline static void im_mat_mem_alloc(IplImage **arg_img, CvSize arg_size, int arg_depth, int arg_channels)
{
	if (*arg_img != NULL)
		return;
	*arg_img = cvCreateImage(arg_size, arg_depth, arg_channels);
	if (*arg_img == NULL)
	{
		fprintf(stderr, "Error!!! unable to allocate memory to image.\n");
		exit(-1);
	}
}

int main(int argc, char **argv)
{
	//create a handle for the video
	CvCapture *video_in = cvCaptureFromFile("C:\\Users\\Nikhil Nayak\\Downloads\\test.avi");

	//error here means error in either opening file or file codec format
	if (video_in == NULL)
	{
		fprintf(stderr, "Error in opening file. Check for path and/or codec.\n");
		return -1;
	}

	//Query 
	cvQueryFrame(video_in);

	//The following section is for obtaining the properties of the video file
	CvSize frame_size_details;
	frame_size_details.height = cvGetCaptureProperty(video_in, CV_CAP_PROP_FRAME_HEIGHT);
	frame_size_details.width = cvGetCaptureProperty(video_in, CV_CAP_PROP_FRAME_WIDTH);

	//Declare variable to hold the count of frames
	long frame_count;

	//The following command skips to the end of the video file
	cvSetCaptureProperty(video_in, CV_CAP_PROP_POS_AVI_RATIO, 1.0);
	//once at the end we can read the AVI colour_frame_in details
	frame_count = (int)cvGetCaptureProperty(video_in, CV_CAP_PROP_POS_FRAMES);
	//now return to the start
	cvSetCaptureProperty(video_in, CV_CAP_PROP_POS_FRAMES, 0.0);
	/*Create 3 windows named "colour_frame_in N" "colour_frame_in N+1" and "Optical Flow" for visualising the output.
	  Make their size match the output autonomously
	*/
	cvNamedWindow("OPTICAL FLOW", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("my_templar", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("frame_n_gray", CV_WINDOW_AUTOSIZE);
	long present_frame = 0;
	double thresh = 0.04;

	while (true)
	{
		static IplImage *colour_frame_in = NULL;
		static IplImage *colour_frame_out = NULL;
		static IplImage *frame_n_gray = NULL;
		static IplImage *frame_n1_gray = NULL;
		static IplImage *image_eig = NULL;
		static IplImage *image_temp = NULL;
		static IplImage *pyramid_n = NULL;
		static IplImage *pyramid_n1 = NULL;
		static IplImage *my_templar = NULL;					//testing purspose only

		//go to the required colour_frame_in. Note multiple frames could be queried in the loop. The very frist position is set outside 
		//the loop
		cvSetCaptureProperty(video_in, CV_CAP_PROP_POS_FRAMES, present_frame);
		colour_frame_in = cvQueryFrame(video_in);
		if (colour_frame_in == NULL)
		{
			fprintf(stderr, "Error getting current colour_frame_in. Reached a premature end.\n");
		}

		//convert to a grayscale image
		im_mat_mem_alloc(&frame_n_gray, frame_size_details, IPL_DEPTH_8U, 1);

		//flip the image vertically. This is done because opencv reads avi frames upside down by default
		cvConvertImage(colour_frame_in, frame_n_gray, /*CV_LOAD_IMAGE_UNCHANGED*/CV_CVTIMG_FLIP);

		//make a full colour backup of the above colour_frame_in to draw on it
		im_mat_mem_alloc(&colour_frame_out, frame_size_details, IPL_DEPTH_8U, 3);

		cvConvertImage(colour_frame_in, colour_frame_out, /*CV_LOAD_IMAGE_UNCHANGED*/CV_CVTIMG_FLIP);

		//obtain the next colour_frame_in
		colour_frame_in = cvQueryFrame(video_in);
		if (colour_frame_in == NULL)
		{
			fprintf(stderr, "Error obtaining colour_frame_in.\n");
			return -1;
		}

		//allocate memory for frame_n1_gray

		im_mat_mem_alloc(&frame_n1_gray, frame_size_details, IPL_DEPTH_8U, 1);
		cvConvertImage(colour_frame_in, frame_n1_gray, /*CV_LOAD_IMAGE_UNCHANGED*/CV_CVTIMG_FLIP);

		//allocating memory for additional storage
		im_mat_mem_alloc(&image_eig, frame_size_details, IPL_DEPTH_32F, 1);
		im_mat_mem_alloc(&image_temp, frame_size_details, IPL_DEPTH_32F, 1);

		//storage array for features in colour_frame_out
		CvPoint2D32f frame1_features[400];
		CvPoint2D32f frame2_features[400];

		int feature_count = 400;

		//perform an edge extraction using the Laplace mask and High Boost Filtering
		//uncomment the following section for using laplacian image intensity averaging for adaptive thresholding
		Mat frame_laplace_in = cvarrToMat(frame_n_gray);
		// frame_n_gray_holder = cvarrToMat(frame_n_gray);
		Mat frame_laplace_out;
		Laplacian(frame_laplace_in, frame_laplace_out, CV_8U, 3, 1, 0, BORDER_DEFAULT);
		uint8_t *mat_accss = frame_laplace_out.data;
		int cols = frame_laplace_out.cols;
		int rows = frame_laplace_out.rows;
		int stride = frame_laplace_out.step;
		long mean_int = 0;
		int tot_elems = rows*cols;

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				mean_int += *(mat_accss + (i*stride + j));
			}
		}
		mean_int /= tot_elems;
		if(mean_int > 20)
			thresh = 0.1;
		if (mean_int >= 10 && mean_int <= 20)
			thresh = 0.04;
		else
			thresh = 0.004;
		my_templar = cvCloneImage(&(IplImage)frame_laplace_out);
		printf("thresh is = %f and Mean intensity = %ld\n", thresh, mean_int);


		//uncomment the following section to use high bosst filtered images for feature detection
		/*
				//convertScaleAbs(frame_laplace_out, frame_laplace_out);
				addWeighted(frame_laplace_in, 1.0, frame_laplace_out, 0.5, 0, frame_laplace_in);
				//my_templar = cvCloneImage(&(IplImage)frame_laplace_in);
				frame_n_gray = cvCloneImage(&(IplImage)frame_laplace_in);
		*/


		/*
		//smoothening filter
		Mat frame_n_gray_holder = cvarrToMat(frame_n_gray);
		Mat my_temp;
		GaussianBlur(frame_n_gray_holder, my_temp, cv::Size(0, 0), 5);
		frame_n_gray = cvCloneImage(&(IplImage)frame_n_gray_holder);

		//sharpen the compare frames

		Mat frame_n_gray_holder = cvarrToMat(frame_n_gray);
		Mat my_temp;
		GaussianBlur(frame_n_gray_holder, my_temp, cv::Size(0, 0), 5);
		addWeighted(frame_n_gray_holder, 1.5, my_temp, -0.5, 0, my_temp);

		frame_n_gray	= cvCloneImage(&(IplImage)frame_n_gray_holder);

		Mat frame_n1_gray_holder = cvarrToMat(frame_n_gray);
		GaussianBlur(frame_n1_gray_holder, my_temp, cv::Size(0, 0), 3);
		addWeighted(frame_n1_gray_holder, 1.5, my_temp, -0.5, 0, my_temp);

		frame_n1_gray = cvCloneImage(&(IplImage)frame_n1_gray_holder);
*/
//This is the shi and tomasi algorithm
//frame_n_gray is the input colour_frame_in
//image_eig and image_temp are temp storage to work on the image
//sixth argument 0.01 (based on the eigen value) - the min quality of the features.
//seventh argument 0.01 - min euclidean distance between features.
//NULL argument towards the end means the function points to the entire input image
//the algorithm returns feature points in frame1_features and the number of features in feature_count which will
//be a quantity less than 400
//use a block size of 5 for calculating the derivative covariance matrix
//penultimate argument enables harris corner detector
//0.04 parameter for the harris corner detector

		cvGoodFeaturesToTrack(frame_n_gray, image_eig, image_temp, frame1_features, &feature_count, 0.01, 0.01, NULL, 3, 1, thresh);
		//cvGoodFeaturesToTrack(frame_n1_gray, image_eig, image_temp, frame2_features, &feature_count, 0.0001, 0.1, NULL, 5, 1, 0.04);

		//run the pyramidal KLT

		//the ith element is not 0 iff we're able to find the same feature in both the frames

		char OF_feature[400];

		//in the following array we store the error associated with the ith feature

		float OF_feature_error[400];

		//define the window or mask size
		CvSize OF_window = cvSize(3, 3);

		//condition for termination
		//no. of iteration shouldn't exceed 20 or epsilon value of 0.3 or better is achieved

		CvTermCriteria optical_flow_termination_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3);

		//allocate memory to pyramid vars for klt

		im_mat_mem_alloc(&pyramid_n, frame_size_details, IPL_DEPTH_8U, 1);
		im_mat_mem_alloc(&pyramid_n1, frame_size_details, IPL_DEPTH_8U, 1);

		cvCalcOpticalFlowPyrLK(frame_n_gray, frame_n1_gray, pyramid_n, pyramid_n1, frame1_features, frame2_features, feature_count, OF_window, 5, OF_feature, OF_feature_error, optical_flow_termination_criteria, 0);


		//draw the flow field

		for (int i = 0; i < feature_count; i++)
		{
			//if no features are found then skip
			if (OF_feature[i] == 0)
				continue;
			int line_thickness = 1;

			//assign line components

			CvScalar draw_line = CV_RGB(255, 55, 55);
			CvPoint point_a;
			CvPoint point_b;
			point_a.x = (int)frame1_features[i].x;
			point_a.y = (int)frame1_features[i].y;
			point_b.x = (int)frame2_features[i].x;
			point_b.y = (int)frame2_features[i].y;

			//calc magnitude and angle of the motion vector of each feature
			double angle = atan2((double)point_a.y - point_b.y, (double)point_a.x - point_b.x);
			double hypotenuse = sqrt(pow_2(point_a.y - point_b.y) + pow_2(point_a.x - point_b.x));

			//arrow lengthening to represent in a legible manner on the ouput image
			point_b.x = (int)(point_a.x - 0.5 * hypotenuse * cos(angle));
			point_b.y = (int)(point_a.y - 0.5 * hypotenuse * cos(angle));

			//draw the line
			cvLine(colour_frame_out, point_a, point_b, draw_line, line_thickness, CV_AA, 0);

			//scale the arrows tips and draw the arrows on the images
			point_a.x = (int)(point_b.x + 9 * cos(angle + stored_pi / 4));
			point_a.y = (int)(point_b.y + 9 * sin(angle + stored_pi / 4));
			cvLine(colour_frame_out, point_a, point_b, draw_line, line_thickness, CV_AA, 0);
			point_a.x = (int)(point_b.x + 9 * cos(angle - stored_pi / 4));
			point_a.y = (int)(point_b.y + 9 * sin(angle - stored_pi / 4));
			cvLine(colour_frame_out, point_a, point_b, draw_line, line_thickness, CV_AA, 0);
		}
		//flip the image around x = 0 y > 0 and x&y < 0
		cvFlip(colour_frame_out, colour_frame_out, 0);
		cvShowImage("OPTICAL FLOW", colour_frame_out);
		cvFlip(my_templar, my_templar, 0);
		cvShowImage("my_templar", my_templar);
		cvFlip(frame_n_gray, frame_n_gray, 0);
		cvShowImage("frame_n_gray", frame_n_gray);
		//uncomment the line below and comment the next section for continuous mode
		//waitKey(10);
		//##FRAMEWISE_START - the following if else mode is for framewise mode  comment the previous line if using this
		
		//wait for key press
		int key_capture;
		key_capture = cvWaitKey(0);

		//if the key pressed is b or B then go to the previous colour_frame_in

		if (key_capture == 'b' || key_capture == 'B')
			present_frame--;
		else if (key_capture == 's' || key_capture == 'S')
			present_frame = present_frame + 100;
		else if (key_capture == 'a' || key_capture == 'A')
			present_frame = present_frame - 100;
		else
			//##FRAMEWISE_END
		
			present_frame = present_frame + 1;
		if (present_frame < 0)
			present_frame = 0;
		if (present_frame >= frame_count - 1)
			present_frame = frame_count - 2;

	}

}