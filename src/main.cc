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

float conv_weight[32768];
float cent[32768];
float WPCA_w[134217728];
float WPCA_b[4096];

float f_result[294912];
float after_softmax[36864];
float vlad[32768];
float fin[4096];

void ListImages(string const &path, vector<string> &images)
{
	images.clear();
	struct dirent *entry;

	/*Check if path is a valid directory path. */
	struct stat s;
	lstat(path.c_str(), &s);
	if (!S_ISDIR(s.st_mode))
	{
		fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
		exit(1);
	}

	DIR *dir = opendir(path.c_str());
	if (dir == nullptr)
	{
		fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
		exit(1);
	}

	while ((entry = readdir(dir)) != nullptr)
	{
		if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN)
		{
			string name = entry->d_name;
			string ext = name.substr(name.find_last_of(".") + 1);
			if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") || (ext == "PNG") || (ext == "png"))
			{
				images.push_back(name);
			}
		}
	}

	closedir(dir);
}

void out_file(DPUTask *task)
{
	int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
	int8_t *result = new int8_t[num];
	cout << num << endl;
	/*dpuGetOutputTensorInHWCInt8(task, OUTPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile("result_HWC.txt", ios::out);
	if(!outfile) {
		cerr<<"open outfile erro"<<endl;
		exit(1);
	}
	for(int i=0; i<num; i++) {
		outfile<<(+result[i])*2<<" ";
	}
	outfile.close();*/

	dpuGetOutputTensorInCHWInt8(task, OUTPUT_NODE, result, num);
	//result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
	ofstream outfile1("result_CHW.txt", ios::out);
	if (!outfile1)
	{
		cerr << "open outfile erro" << endl;
		exit(1);
	}
	for (int i = 0; i < num; i++)
	{
		result[i] = (+result[i]);
	}
	for (int i = 0; i < num; i++)
	{
		outfile1 << result[i] * 2 << " ";
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

void normalize(float *data)
{
	for (int i = 0; i < 576; i++)
	{
		float sum = 0;
		for (int j = 0; j < 512; j++)
		{
			sum = sum + data[j * 576 + i] * data[j * 576 + i];
		}
		sum = sqrt(sum);
		if (sum < 1e-12)
			sum = 1e-12;
		for (int j = 0; j < 512; j++)
		{
			data[j * 576 + i] = data[j * 576 + i] / sum;
		}
	}
}

void conv(float *data, float *a, float *w)
{
	for (int i = 0; i < 36864; i++)
	{
		a[i] = 0;
	}
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 576; j++)
		{
			for (int k = 0; k < 512; k++)
			{
				a[i * 576 + j] = a[i * 576 + j] + data[k * 576 + j] * w[512 * i + k];
			}
		}
	}
	for (int i = 0; i < 576; i++)
	{
		float sum = 0;
		for (int j = 0; j < 64; j++)
		{
			a[j * 576 + i] = a[j * 576 + i] - 360;
			sum = sum + exp(a[j * 576 + i]);
		}
		for (int j = 0; j < 64; j++)
		{
			a[j * 576 + i] = exp(a[j * 576 + i]) / sum;
		}
	}
}

void vlad_core(float *data, float *a, float *v, float *c)
{
	float *temp = new float[64 * 512 * 576];
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			for (int k = 0; k < 576; k++)
			{
				temp[i * 576 * 512 + j * 576 + k] = data[j * 576 + k] + c[i * 512 + j];
				temp[i * 576 * 512 + j * 576 + k] *= a[576 * i + k];
			}
		}
	}
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			for (int k = 0; k < 576; k++)
			{
				v[i * 512 + j] += temp[i * 512 * 576 + j * 576 + k];
			}
		}
	}
}

void normalize2(float *vlad)
{
	float *temp = new float[32768];
	for (int i = 0; i < 64; i++)
	{
		float sum = 0;
		for (int j = 0; j < 512; j++)
		{
			sum = sum + vlad[j + i * 512] * vlad[j + i * 512];
		}
		sum = sqrt(sum);
		if (sum < 1e-12)
			sum = 1e-12;
		for (int j = 0; j < 512; j++)
		{
			temp[j * 64 + i] = vlad[j + i * 512] / sum;
		}
	}
	float sum = 0;
	for (int i = 0; i < 32768; i++)
		sum += (temp[i] * temp[i]);
	sum = sqrt(sum);
	for (int i = 0; i < 32768; i++)
		vlad[i] = (temp[i] / sum);
}

void fc(float *v, float *f, float *w, float *b)
{
	for (int i = 0; i < 4096; i++)
		f[i] = 0;
	for (int i = 0; i < 4096; i++)
	{
		for (int j = 0; j < 32768; j++)
		{
			f[i] = f[i] + v[j] * w[i * 32768 + j];
		}
		f[i] = f[i] + b[i];
	}
}

void normalize3(float *f)
{
	float sum = 0;
	for (int i = 0; i < 4096; i++)
	{
		sum += f[i] * f[i];
	}
	sum = sqrt(sum);
	for (int i = 0; i < 4096; i++)
	{
		f[i] = f[i] / sum;
	}
}

void run_netvlad(DPUTask *task, float *conv_w, float *cent, float *WPCA_w, float *WPCA_b)
{
	assert(task);
	vector<string> images;
	ListImages(baseImagePath, images);
	if (images.size() == 0)
	{
		cerr << "\nError: Not images exist in " << baseImagePath << endl;
		return;
	}

	for (auto &imageName : images)
	{
		cout << "\nLoad image : " << imageName << endl;
		Mat image = imread(baseImagePath + imageName);
		float mean[3] = {0, 0, 0};
		//cvtColor(image, image, );
		dpuSetInputImage(task, INPUT_NODE, image, mean);

		/* Launch VGGfae CONV Task */
		cout << "\nRun netvlad ..." << endl;
		dpuRunTask(task);

		/* Get DPU execution time (in us) of CONV Task */
		long long timeProf = dpuGetTaskProfile(task);
		cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";

		int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
		int8_t *result = new int8_t[num];
		dpuGetOutputTensorInCHWInt8(task, OUTPUT_NODE, result, num);
		for (int i = 0; i < num; i++)
		{
			f_result[i] = (+result[i]) * 2;
		}
		delete[] result;

		normalize(f_result);
		conv(f_result, after_softmax, conv_w);
		for (int i = 0; i < 32768; i++)
			vlad[i] = 0;
		vlad_core(f_result, after_softmax, vlad, cent);

		normalize2(vlad);

		fc(vlad, fin, WPCA_w, WPCA_b);
		normalize3(fin);

		//cout<< fin[0] << " " << fin[1] << " " << fin[4095]<<endl;

		//out_file(task);
	}
}

int main(void)
{

	// Attach to DPU driver and prepare for running
	dpuOpen();

	// Load DPU Kernel for DenseBox neural network
	DPUKernel *kernel = dpuLoadKernel("netvlad");

	DPUTask *task;
	task = dpuCreateTask(kernel, 0);

	//load vlad parameters
	fstream in("/home/linaro/netvlad/model/vlad_weight", ios::in | ios::binary);
	in.read((char *)conv_weight, 32768 * sizeof(float));
	in.read((char *)cent, 32768 * sizeof(float));
	in.read((char *)WPCA_w, 134217728 * sizeof(float));
	in.read((char *)WPCA_b, 4096 * sizeof(float));
	in.close();

	// Doing face detection.
	run_netvlad(task, conv_weight, cent, WPCA_w, WPCA_b);

	dpuDestroyTask(task);

	// Destroy DPU Kernel & free resources
	dpuDestroyKernel(kernel);

	// Dettach from DPU driver & release resources
	dpuClose();

	return 0;
}
