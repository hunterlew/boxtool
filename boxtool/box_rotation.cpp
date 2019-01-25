#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

const double pi = 3.14159265357;

const float map_range_length = 100.0;
const float map_range_width = 50.0;
const float map_scale = 0.2;


struct BoxObject
{
	float x;
	float y;
	float box_length;
	float box_width;
	float yaw;
	cv::Scalar color;
};


static cv::Point map2pixel(float x, float y)
{
	return cv::Point(y / map_scale + map_range_width / map_scale,
		map_range_length / map_scale - x / map_scale);
}

static cv::Point map2pixel(cv::Point2f pt)
{
	return map2pixel(pt.x, pt.y);
}

static void box_rot_new(float yaw, std::vector<cv::Point2f> &v)
{
	assert(v.size() == 4);
	
	yaw *= pi / 180;
	Eigen::Quaterniond q(cos(yaw/2), 0, 0, sin(yaw/2));

	for (auto &pt : v)
	{
		Eigen::Vector3d vec(pt.x, pt.y, 0);
		vec = q * vec;
		pt.x = vec(0);
		pt.y = vec(1);
	}
}

static void box_rot(float yaw, std::vector<cv::Point2f> &v)
{
	assert(v.size() == 4);

	std::cout << "\nbefore rot: \n";
	for (auto pt : v)
	{
		std::cout << pt << ",";
	}

	// v {minx, miny, maxx, maxy}
	Eigen::MatrixXd corner(2,4);  // left top, left down, right down, right top
	corner << v[0].x, v[1].x, v[2].x, v[3].x,
					  v[0].y, v[1].y, v[2].y, v[3].y;

	Eigen::Matrix2d R;
	yaw *= pi / 180;
	R << cos(yaw), -sin(yaw),
		     sin(yaw), cos(yaw);

	corner = R * corner;

	v[0].x = corner(0, 0);
	v[0].y = corner(1, 0);
	v[1].x = corner(0, 1);
	v[1].y = corner(1, 1);
	v[2].x = corner(0, 2);
	v[2].y = corner(1, 2);
	v[3].x = corner(0, 3);
	v[3].y = corner(1, 3);

	std::cout << "\nafter rot: \n";
	for (auto pt : v)
	{
		std::cout << pt << ",";
	}
}

static void draw_box(cv::Mat &m, BoxObject obj)
{
	float x = obj.x, y = obj.y;
	float box_length = obj.box_length, box_width = obj.box_width;
	float yaw = obj.yaw;
	cv::Scalar color = obj.color;

	std::vector<cv::Point2f> corner(4);
	corner[0].x = box_length / 2;
	corner[0].y = -box_width / 2;
	corner[1].x = -box_length / 2;
	corner[1].y = -box_width / 2;
	corner[2].x = -box_length / 2;
	corner[2].y = box_width / 2;
	corner[3].x = box_length / 2;
	corner[3].y = box_width / 2;
	// box_rot(yaw, corner);
	box_rot_new(yaw, corner);
	for (auto &pt : corner)
	{
		pt.x += x;
		pt.y += y;
	}

	cv::circle(m, map2pixel(x, y), 2, color, 2);
	cv::line(m, map2pixel(corner[0]), map2pixel(corner[1]), color, 2);
	cv::line(m, map2pixel(corner[0]), map2pixel(corner[3]), color, 2);
	cv::line(m, map2pixel(corner[2]), map2pixel(corner[1]), color, 2);
	cv::line(m, map2pixel(corner[2]), map2pixel(corner[3]), color, 2);
	cv::line(m, map2pixel(x, y), map2pixel((corner[0] + corner[3]) * 0.5), color, 2);
}

static void draw_local_map(cv::Mat &m)
{
	BoxObject origin{ 0, 0, 4, 2, 0, cv::Scalar(255,255,255) };
	draw_box(m, origin);
}


int main()
{
	cv::Mat lm(map_range_length*2 / map_scale, map_range_width*2 / map_scale, CV_8UC3, cv::Scalar(0, 0, 0));
	draw_local_map(lm);

	BoxObject obj1{ 50, -5, 15, 4, 0, cv::Scalar(0,255,0) };
	draw_box(lm, obj1);

	BoxObject obj2{ 20, 17, 4, 2, 30, cv::Scalar(0,255,0) };
	draw_box(lm, obj2);

	BoxObject obj3{ -30, -25, 12, 3, -135, cv::Scalar(0,255,0) };
	draw_box(lm, obj3);

	cv::namedWindow("lm");
	cv::imshow("lm", lm);
	cv::imwrite("box_rotation_example.png", lm);
	cv::waitKey(-1);
}
