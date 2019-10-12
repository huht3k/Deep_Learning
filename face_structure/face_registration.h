#pragma once

#ifndef FACE_REGISTRATION_H
#define FACE_REGISTRATION_H

#include<iostream>
#include <string>
#include <time.h>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <sys/timeb.h>

// for concurrency
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <deque>

#include "math_functions.h"
#include "common.h"
#include "math.h"
#include "time.h"
#include "ctime"

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

// opencv
#include "opencv2\core.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"

extern bool bFrontWorking;

// seeta
#include "face_identification.h"
#include "face_detection.h"
#include "face_alignment.h"

#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "


// third party
#include "json/json.h"
#pragma comment(lib, "../../x64/release/JsonCPP.lib")

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// Mr. Hu: added 
#define FEAT_SIZE 2048
#define FRONT_FACES_FULL  3
#define HOME 0

// 1: me.score, me.norm2, me.feat[2048]
extern float FEATS[(2 + FEAT_SIZE) * 10000 * sizeof(float)];
extern float Register_Threshold;


extern std::string DATA_DIR;
extern std::string MODEL_DIR; 
extern std::string FRONT_DIR;
extern std::string DB_DIR;
extern std::string DB_FT_DIR;
extern std::string DB_FACE_DIR;
extern std::string FRONT_JSON_DIR;

typedef struct _id {
	std::string m_FaceId;
	_id(std::string str) {
		m_FaceId = str;
	}
	_id() {};
	_id(long id) {
		m_FaceId = std::to_string(id);
	}
	std::string& asString() {
		return m_FaceId;
	}
}Face_ID;

typedef struct _myface
{
public:
	bool hasFace = false;
	cv::Mat color_img;
	cv::Mat gray_img;
	float feat[FEAT_SIZE];
	float score = -1.0f;
	float norm2 = 0.0f;
	// x, y for the facialLandmark
	seeta::FacialLandmark landmarks[5];
	void getFeat(float* pft)
	{
		memcpy(&feat[0], pft, FEAT_SIZE * sizeof(float));
	}
	struct _myface& operator = (seeta::FacialLandmark *pt5)
	{
		for (int i = 0; i < 5; i++)
		{
			this->landmarks[i].x = pt5[i].x;
			this->landmarks[i].y = pt5[i].y;
		}
		return *this;
	}
}MyFace;

typedef struct _landmarks {
	seeta::FacialLandmark points[5];
	_landmarks(seeta::FacialLandmark pts[5]) {
		for (int i = 0; i < 5; i++) {
			points[i].x = pts[i].x;
			points[i].y = pts[i].y;
		}
	}
} MyLandmarks;

std::string getTime();
void DrawLandmarks(cv::Mat img, seeta::FacialLandmark points[5]);
void DetectLandmarks(std::vector<MyLandmarks>& landmarks, seeta::FaceAlignment& point_detector,
	seeta::ImageData& img_data, std::vector<seeta::FaceInfo>& faces);
void RecFrame(cv::Mat img, cv::Rect face_rect, bool bestFace);
void FindBestFace(std::vector<seeta::FaceInfo> faces, float& score, int& best_face);
long GetFileNames(const std::string& dir, std::vector<std::string>& filenames);
void CopyFile(const std::string& from, const std::string& to);
void DeleteFile(const std::string& file);
void ParseJsonFile(std::string& oneJsonFile, Json::Value& jsonRoot);
bool isBlurred(cv::Mat& grayImg, double& var, double threshold = 10.0);
void myPutText(cv::Mat img, std::string text, int font_face = cv::FONT_HERSHEY_COMPLEX,
	double font_scale = 2, int thickness = 2);


class UpdateDB {
public:
	bool isFirstTime(std::string& oneFrontJsonFile);
	UpdateDB() {};
	UpdateDB(std::string& db_dir, std::string& front_dir, 
		std::string& db_feat_dir, std::string& db_face_dir);
	~UpdateDB();

	void DataMaking();
	void VisionFaces();
	
	long initLoadBinData();
	void CreateBinFiles();
	void CreateIDFile();
	void FaceDsp(MyFace& me, cv::Mat& src_img, cv::Mat& img_gray, float score, float* feats,
		std::vector<MyLandmarks>& landmarks, int best_face, float SimThreshold,
		seeta::FaceIdentification& face_recognizer, bool& isNewJsonFile, int cnt);
	void Write2BinFile(MyFace& me, seeta::FaceIdentification& face_recognizer,
		std::string& time_stamp, std::string& face_name, std::string& feat_name);
	void Write2DbJsonFile(Json::Value& jsonRoot, std::string& json_name);
	void UpdateDB::Write2FrontJsonFile(std::string& time_stamp, std::string& face_name,
		std::string& feat_name);
	void GenerateDbJsonFileName(std::string& json_name);   // not used
	void TrashFrontBin();
	bool isQNotEmpty();
	bool isThereDbJsonFiles();
	void Show();
	int getQSize();

private:
	bool _isSimLgTh();   // is sim larger than the threshold
	void _LoadFeats();
	void _AppendFeats();
	void _UpdateFeats();

	std::string m_DbJsonDir;
	std::string m_DbFeatDir;
	std::string m_DbFaceDir;
	std::string m_DbJsonFile;
	std::string m_DbFeatFile;
	std::string m_DbFaceFile;
	std::string m_FrontJsonDir;
	std::string m_FrontFeatFile;
	std::string m_FrontFaceFile;
	std::string m_FrontJsonFile;
	std::vector<std::string> m_DbJsonFiles;
	std::vector<std::string> m_FrontJsonFiles;

	long m_DbSize;
	long m_FrontSize;
	long m_FeatPos;    // where new feats to be added to in global FEATS[]
	long m_FeatNum;    // total number of the feat sets or the face number in the db
	bool m_EmptyDB;

	float m_FrontScore;
	float m_DbScore;
	float m_FrontNorm2;
	float m_DbNorm2;
	float m_Sim;
	float* m_Feat;
	float m_Threshold;

	std::string m_TimeStamp;

	static long m_CNT;  //the number of faces in the Db

	Face_ID m_FaceId;
	long m_AppearTimes;
	long m_Who;   // from beginning at 0 not 1

	std::mutex m_Mu;
	std::mutex m_MuShow;
	std::deque<std::string> m_Que;
	std::condition_variable m_CondVar;
	std::condition_variable m_CondShow;

	bool m_Show;
};

#endif