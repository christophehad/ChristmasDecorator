#pragma once
using namespace cv;
using namespace std;

// Short for Christmas Decorator
namespace CD {
	// directory of source files, where /data should be
	const string srcDir = "../";

	// shared paths
	const string inputImagePath = srcDir + "/data/cmp_b0377.jpg";
	const string inputLabelsPath = srcDir + "/data/cmp_b0377.png";

	// colors
	const Vec3b windowColor = { 255, 85, 0 };
}
