/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 2.05
 *
 * Description:
 * Sample source code showing how to deploy DenseBox neural network on
 * DeePhi DPU.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

#define INPUT_NODE "ConvNdBackward1"
#define OUTPUT_NODE "ConvNdBackward29"
#define length 4096

using namespace std;
//using namespace std::chrono;
using namespace cv;

const string baseImagePath = "/home/linaro/netvlad/test_image/";

void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
	exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
	exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
			string name = entry->d_name;
			string ext = name.substr(name.find_last_of(".") + 1);
			if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
				images.push_back(name);
			}
		}
    }

    closedir(dir);
}

void out_file(DPUTask* task) {
	int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
	int8_t* result = new int8_t[num];
	cout<<num<<endl;
	dpuGetOutputTensorInHWCInt8(task, OUTPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile("result_HWC.txt", ios::out);
	if(!outfile) {
		cerr<<"open outfile erro"<<endl;
		exit(1);
	}
	for(int i=0; i<num; i++) {
		outfile<<(+result[i])*2<<" ";
	}
	outfile.close();

	dpuGetOutputTensorInCHWInt8(task, OUTPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile1("result_CHW.txt", ios::out);
	if(!outfile1) {
		cerr<<"open outfile erro"<<endl;
		exit(1);
	}
	for(int i=0; i<num; i++) {
		result[i]=(+result[i]);
	}
	for(int i=0; i<num; i++) {
		outfile1<<result[i]*2<<" ";
	}
	outfile1.close();
	delete[] result;
}

/*void out_file2(DPUTask* task) {
	int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
	float* result = new float[num];
	dpuGetOutputTensorInHWCFP32(task, INPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile("input_HWC.txt", ios::out);
	if(!outfile) {
		cerr<<"open outfile erro"<<endl;
		exit(1);
	}
	for(int i=0; i<num; i++) {
		outfile<<result[i]<<" ";
	}
	outfile.close();

	dpuGetOutputTensorInCHWFP32(task, INPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile1("input_CHW.txt", ios::out);
	if(!outfile1) {
		cerr<<"open outfile erro"<<endl;
		exit(1);
	}
	for(int i=0; i<num; i++) {
		outfile1<<result[i]<<" ";
	}
	outfile1.close();
	delete[] result;
}*/

void run_netvlad(DPUTask *task) {
	assert(task);
	vector<string> images;
	ListImages(baseImagePath, images);
	if (images.size() == 0) {
        	cerr << "\nError: Not images exist in " << baseImagePath << endl;
        	return;
       	}

	for(auto &imageName : images) {
    	cout<< "\nLoad image : " << imageName << endl;
		Mat image = imread(baseImagePath + imageName);
		float mean[3]={0,0,0};
		//cvtColor(image, image, );
		dpuSetInputImage(task, INPUT_NODE, image, mean);

		/* Launch VGGfae CONV Task */
		cout << "\nRun netvlad ..." << endl;
		dpuRunTask(task);

		/* Get DPU execution time (in us) of CONV Task */
		long long timeProf = dpuGetTaskProfile(task);
		cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";
		
		out_file(task);
    }
	
	
    
}

int main(void) {
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Load DPU Kernel for DenseBox neural network
    DPUKernel *kernel = dpuLoadKernel("netvlad");

	DPUTask *task;
	task = dpuCreateTask(kernel, 0);

    // Doing face detection.
    run_netvlad(task);

	dpuDestroyTask(task);

    // Destroy DPU Kernel & free resources
    dpuDestroyKernel(kernel);

    // Dettach from DPU driver & release resources
    dpuClose();

    return 0;
}
