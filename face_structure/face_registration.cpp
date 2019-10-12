#include "face_registration.h"

bool bWorking = true;
// 1: me.score, me.norm2, me.feat[2048]
float FEATS[(2 + FEAT_SIZE) * 10000 * sizeof(float)];
float Register_Threshold = 0.7;
#ifdef _WIN32
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#if HOME
std::string FRONT_DIR = "H:\\MetroData\\FrontData";
std::string DB_DIR = "H:\\MetroData\\BackData\\Json";
std::string DB_FT_DIR = "H:\\MetroData\\BackData\\BinDescript";
std::string DB_FACE_DIR = "H:\\MetroData\\BackData\\Faces";
std::string FRONT_JSON_DIR = "H:\\MetroData\\FrontData\\Json";
#else
std::string FRONT_DIR = "G:\\\MetroData\\FrontData";
std::string DB_DIR = "G:\\MetroData\\BackData\\Json";
std::string DB_FT_DIR = "G:\\MetroData\\BackData\\BinDescript";
std::string DB_FACE_DIR = "G:\\MetroData\\BackData\\Faces";
std::string FRONT_JSON_DIR = "G:\\MetroData\\FrontData\\Json";
#endif
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
std::string FRONT_DIR = "/home/huht/FrontData";
#endif


std::string getTime() {
	time_t timep;
	char tmp[64];
	time(&timep);
	strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&timep));

	// for ms
	char msTmp[100];
	struct timeb start;
	ftime(&start);
	sprintf(msTmp, "_%03u", start.millitm);
	strcat(tmp, msTmp);

	return tmp;
}

// Visualize the landmarks
void DrawLandmarks(cv::Mat img, seeta::FacialLandmark points[5]) {
	for (int i = 0; i < 5; i++)
	{
		cv::circle(img, cv::Point(points[i].x, points[i].y), 2,
			CV_RGB(0, 255, 0), 4, 8, 0);
	}
}

void RecFrame(cv::Mat img, cv::Rect face_rect, bool bestFace) {

	if (bestFace)
		cv::rectangle(img, face_rect, CV_RGB(255, 0, 0), 4, 8, 0);
	else
		cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
}

// find the best score face
void FindBestFace(std::vector<seeta::FaceInfo> faces, float& score, int& best_face) {
	int num_face = faces.size();
	for (int i = 0; i < num_face; i++) {
		if (faces[i].score > score) {
			score = faces[i].score;
			best_face = i;
		}
	}
}

long GetFileNames(const std::string& dir,
	std::vector<std::string>& filenames) {
	fs::path path(dir);
	if (!fs::exists(path)) {
		std::cout << dir << " not exist" << std::endl;
		exit(0);
	}
	fs::directory_iterator end_iter;
	for (fs::directory_iterator iter(path); iter != end_iter; iter++) {
		if (fs::is_regular_file(iter->status())) {
			filenames.push_back(iter->path().string());
		}
		// recursive
		if (fs::is_directory(iter->status())) {
			GetFileNames(iter->path().string(), filenames);
		}
	}
	return filenames.size();
}

// from should be a regular file 
// to should be a directory or a file
void CopyFile(const std::string& from, const std::string& to) {
	const fs::path src(from);
	const fs::path dst(to);
	// from should be a regular file
	if (!fs::exists(src)) {
		std::cout << from << " not exists" << std::endl;
		exit(0);
	}
	if (fs::is_directory(src)) {
		std::cout << from << " is a directory" << std::endl;
		std::cout << "the file to be copied should be a regular file" << std::endl;
		exit(0);
	}
	// to should be a directory or file
	if (fs::exists(dst.parent_path()) && !fs::is_directory(dst)) {
		fs::copy_file(src, dst, fs::copy_option::overwrite_if_exists);
	}
	else if (!fs::exists(dst)) {
		fs::create_directories(dst);
		fs::copy_file(src, dst / src.filename(), fs::copy_option::overwrite_if_exists);
	}
	else if (fs::is_directory(dst)) {
		fs::copy_file(src, dst / src.filename(), fs::copy_option::overwrite_if_exists);
	}
	else if (!fs::is_regular_file(src)) {
		fs::copy_file(src, dst, fs::copy_option::overwrite_if_exists);
	}
	else {
		std::cout << "fail to copy file from " << from << " to " << to << std::endl;
		exit(0);
	}

}

void DeleteFile(const std::string& file) {
	fs::remove(fs::path(file));
}

void ParseJsonFile(std::string& oneJsonFile, Json::Value& jsonRoot) {
	Json::Reader jsonReader;
	std::ifstream ifs(oneJsonFile);
	if (!ifs.is_open()) {
		std::cout << "cannot open file: " << oneJsonFile << std::endl;
		exit(0);
	}
	if (!jsonReader.parse(ifs, jsonRoot)) {
		std::cout << "fail to parse the file: " << oneJsonFile << std::endl;
		exit(0);
	}
	ifs.close();
}

// all the landmarks will be recordedc to vector<> landmarks
void DetectLandmarks(std::vector<MyLandmarks>& landmarks, seeta::FaceAlignment& point_detector,
	seeta::ImageData& img_data, std::vector<seeta::FaceInfo>& faces) {
	int32_t num_face = static_cast<int32_t>(faces.size());
	seeta::FacialLandmark points[5];
	for (int k = 0; k < num_face; k++) {
		point_detector.PointDetectLandmarks(img_data, faces[k], points);
		landmarks.push_back(MyLandmarks(points));
	}
}

bool isBlurred(cv::Mat& grayImg, double& var, double threshold) {
	cv::Scalar mean;
	cv::Scalar stddev;
	cv::Mat lapImg;

	cv::Laplacian(grayImg, lapImg, CV_64F);
	//cv::Laplacian(grayImg, lapImg, CV_8U);
	cv::meanStdDev(lapImg, mean, stddev, cv::Mat());
	var = stddev.val[0] * stddev.val[0];
	if (var <= threshold)
		return true;
	else
		return false;
}

void myPutText(cv::Mat img, std::string text, int font_face, 
	double font_scale, int thickness) {
	//设置绘制文本的相关参数
	int baseline;
	//获取文本框的长宽
	cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

	//将文本框居中/右上角绘制
	//将文本框居中/右上角绘制
	cv::Point origin;
	origin.x = img.cols * 4 / 5 - text_size.width / 2;
	origin.y = img.rows / 5 + text_size.height / 2;
	cv::putText(img, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255),
		thickness, 8, 0);
}



// class UpdateDB
long UpdateDB::m_CNT = 0;


//UpdateDB::UpdateDB() {}
UpdateDB::UpdateDB(std::string& db_dir, std::string& front_dir,
	std::string& db_feat_dir, std::string& db_face_dir) {
	m_DbJsonDir = db_dir;
	m_FrontJsonDir = front_dir;
	m_DbFeatDir = db_feat_dir;
	m_DbFaceDir = db_face_dir;
	m_DbSize = GetFileNames(m_DbJsonDir, m_DbJsonFiles);
	/* 
	if (m_DbSize > 0)
		m_DbJsonFile = m_DbJsonFiles.front();
	*/
	m_DbJsonFile = "_NOT_BORN_";
	m_Show = true;
	
	// when loop processing the front cropped data,
	//m_FeatPos, m_CNT, m_Who should be added the previous processing
	if (m_DbSize <= 0) {
		m_EmptyDB = true;
	}
	else {
		m_EmptyDB = false;
	}
	m_FeatNum = m_DbSize;    // init the number of feats or faces in dataBase
	m_FeatPos = 0;    // not begin to load feats from dataBase
	m_CNT = m_DbSize;   // for Id (the used Id from 1 to m_CNT)
	m_Who = 0;

	long db_size = initLoadBinData();
	std::cout << "in construct, db_size = " << db_size << std::endl;
	if (db_size != m_FeatNum) {
		std::cout << "init loading not completed" << std::endl;
		exit(0);
	}

	// front end
	m_FrontSize = GetFileNames(m_FrontJsonDir, m_FrontJsonFiles);
	for (auto i = 0; i < m_FrontSize; i++) {
		m_Que.push_front(m_FrontJsonFiles[i]);
	}
	std::cout << "the size of the front json files : " << m_FrontSize << std::endl;

	m_Threshold = Register_Threshold;
	m_Feat = new float[FEAT_SIZE];

}
UpdateDB::~UpdateDB() {
	delete[] m_Feat;
}

void UpdateDB::VisionFaces() {
	// Initialize face detection model
	seeta::FaceDetection detector(".//model//seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(20);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	// Initialize face alignment model 
	seeta::FaceAlignment point_detector(".//model//seeta_fa_v1.0.bin");
	// Initialize face identification model
	seeta::FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
#if HOME
	std::string DATA_DIR = "H:\\MetroData\\人脸跟踪视频测试集\\";
#else
	std::string DATA_DIR = "G:\\dataset\\人脸跟踪视频测试集\\";
#endif
	std::string video_name = "multi_face.AVI";
	//std::string video_name = "0188_03_021_al_pacino.AVI";
	//std::string video_name = "1034_03_006_jet_li.AVI";

	// n * c * h * w
	int feat_size = face_recognizer.feature_size();
	EXPECT_EQ(feat_size, FEAT_SIZE);
	float* feats = new float[FEAT_SIZE];

	//cv::VideoCapture capture(DATA_DIR + video_name);
	cv::VideoCapture capture(0);
	MyFace me;
	float SimThreshold = 0.8;
	if (!capture.isOpened()) {
		std::cout << "video not open." << std::endl;
		exit(0);
	}

	cv::Mat img;

	bool stop(false);
	bool CROP = false;
	int cnt = 0;
	//long total_time = 0;
	while (!stop) {
		//std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
		if (!capture.read(img))
		{
			std::cout << "no video frame" << std::endl;
			exit(0);
		}
		cv::Mat img_gray;
		if (img.channels() != 1)
			cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		else
			img_gray = img;

		double var;
		if (isBlurred(img_gray, var)) {
			cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
			cv::imshow("Test", img);
			char c = cv::waitKey(1);
			if (c == 27) {
				break;
			}
			continue;
		}
		seeta::ImageData img_data;
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		img_data.num_channels = 1;

		// Mr. Hu: deep copy the img to src_img for the feature extraction
		cv::Mat src_img;
		img.copyTo(src_img);
		seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
		src_img_data.data = src_img.data;

		// detect faces
		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
		cv::Rect face_rect;
		int32_t num_face = static_cast<int32_t>(faces.size());
		// skip current frame
		if (num_face == 0) {
			cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
			cv::imshow("Test", img);
			char c = cv::waitKey(1);
			if (c == 27) {
				break;
			}
			continue;
		}
			
		// find the best score face
		float score = -1.0f;
		int best_face = -1;
		FindBestFace(faces, score, best_face);

		// draw the rectangle on the video frame
		for (int32_t i = 0; i < num_face; i++) {
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;
			RecFrame(img, face_rect, i == best_face);
		}

		// Detect 5 facial landmarks for each detected faces
		std::vector<MyLandmarks> landmarks;     // the detected landmarks
		DetectLandmarks(landmarks, point_detector, img_data, faces);

		// Mr. Hu: feature extraction for the best_face 
		face_recognizer.ExtractFeatureWithCropMy("testing", false,
			src_img_data, landmarks[best_face].points, feats);

		bool isNewJsonFile = false;
		FaceDsp(me, src_img, img_gray, score, feats, landmarks, best_face,
			SimThreshold, face_recognizer, std::ref(isNewJsonFile), cnt);
		if (isNewJsonFile) {
			std::unique_lock<std::mutex> locker(m_Mu);
			//locker.lock();
			m_Que.push_front(m_FrontJsonFile);
			locker.unlock();
			m_CondVar.notify_all();
		}

		for (int k = 0; k < num_face; k++)
			DrawLandmarks(img, landmarks[k].points);

		cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
		cv::imshow("Test", img);
		cnt++;

		//std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
		//auto diff = std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count();
		//std::cout << "Time Taken for one frame = " << diff << " MilliSeconds" << std::endl;
		//total_time += diff;

		char c = cv::waitKey(1);
		if (c == 27)
			break;
	}

	// put the last me to the files
	if (me.hasFace) {
		//std::cout << "haha" << std::endl;
		std::string time_stamp = getTime();
		std::string face_name;
		std::string feat_name;
		Write2BinFile(me, face_recognizer, time_stamp, face_name, feat_name);
		Write2FrontJsonFile(time_stamp, face_name, feat_name);

		std::unique_lock<std::mutex> locker(m_Mu);
		//locker.lock();
		m_Que.push_front(m_FrontJsonFile);
		bWorking = false;
		locker.unlock();
		m_CondVar.notify_all();
	}
	//m_CondVar.notify_all();
	std::unique_lock<std::mutex> locker(m_Mu);
	//locker.lock();
	bWorking = false;
	locker.unlock();
	m_CondVar.notify_all();

	//std::cout << "Total time = " << total_time / 1000 << " seconds" << std::endl;

	cv::destroyAllWindows();
	capture.release();

	//std::cout << "bWorking : " << bWorking << std::endl;
	delete[] feats;
}

void UpdateDB::DataMaking() {
	//std::cout << "bWorking: " << bWorking << std::endl;
	//std::cout << "isQNotEmpty: " << isQNotEmpty() << std::endl;
	std::unique_lock<std::mutex> locker(m_Mu);
	while (bWorking || isQNotEmpty()) {
		//locker.unlock();
		//std::unique_lock<std::mutex> locker(m_Mu);
		m_CondVar.wait(locker, std::bind(&UpdateDB::isQNotEmpty, this));
		m_FrontJsonFile = m_Que.back();
		m_Que.pop_back();
		locker.unlock();
		
		// in the meantime, get m_FrontFeatFile, m_FrontFaceFile
		//                  and m_TimeStamp
		//m_FrontJsonFile = m_FrontJsonFiles[i];
		bool is_first_time = isFirstTime(m_FrontJsonFile);
		if (is_first_time) {
				// in the meantime, define m_DbFaceFile and m_DbFeatFile
				// and m_DbJsonFiles.push_back(json_name);
				CreateIDFile();
				CreateBinFiles();
				_AppendFeats();
			}
		else {      // select the better score and update it
				Json::Value jsonRoot;
				Json::Value jsonItem;
				long appFrqs;
				//if ((m_FrontScore > m_DbScore) && (fabs(m_FrontScore - m_DbScore) > 1e-6)) { 
				if ((m_FrontScore > m_DbScore)) {
					// parse jsonfile
					std::string db_json_file = m_DbJsonFiles[m_Who];
					// delete old bin files
					ParseJsonFile(db_json_file, jsonRoot);
					std::string db_face_file = jsonRoot["faceFile"].asString();
					std::string db_feat_file = jsonRoot["featFile"].asString();
					DeleteFile(db_face_file);
					DeleteFile(db_feat_file);

					//copy m_FrontFeatFile to db and rename it m_DbFeatFile;
					//copy m_FrontFaceFile to db and rename it m_DbFaceFile
					m_DbFaceFile = m_DbFaceDir + "\\ID_" + jsonRoot["Id"].asString() + "_" + m_TimeStamp + ".jpg";
					m_DbFeatFile = m_DbFeatDir + "\\FT_" + m_TimeStamp + ".dat";
					CreateBinFiles();

					//re-write jsonfile
					appFrqs = std::stol(jsonRoot["appFrqs"].asString()) + 1;
					jsonItem = jsonRoot["timeStamp"];
					jsonItem[appFrqs] = m_TimeStamp;
					jsonRoot["timeStamp"] = jsonItem;   // modify time stamps
					jsonRoot["appFrqs"] = std::to_string(appFrqs);   // modify appFrqs
					jsonRoot["faceFile"] = m_DbFaceFile;
					jsonRoot["featFile"] = m_DbFeatFile;
					// delete the db json file and rewrite it
					DeleteFile(db_json_file);
					std::string json_name = m_DbJsonDir + "\\ID_" + jsonRoot["Id"].asString() + "_" + m_TimeStamp + ".json";
					Write2DbJsonFile(jsonRoot, json_name);
					m_DbJsonFiles[m_Who] = json_name;
					//update FEATS
					_UpdateFeats();

					// for show
					std::unique_lock<std::mutex> guardShow(m_MuShow);
					m_DbJsonFile = json_name;
					m_Show = true;
					guardShow.unlock();
					m_CondShow.notify_all();
				}
				else {
					// re-write db json file; only modify the time stamp
					std::string db_json_file = m_DbJsonFiles[m_Who];
					ParseJsonFile(db_json_file, jsonRoot);
					appFrqs = std::stol(jsonRoot["appFrqs"].asString()) + 1;
					std::cout << "appFrqs = " << appFrqs << std::endl;
					jsonItem = jsonRoot["timeStamp"];
					jsonItem[appFrqs] = m_TimeStamp;
					jsonRoot["timeStamp"] = jsonItem;   // modify time stamps
					jsonRoot["appFrqs"] = std::to_string(appFrqs);   // modify appFrqs
																	 // delete the db json file and rewrite it
					DeleteFile(db_json_file);
					std::string json_name = m_DbJsonDir + "\\ID_" + jsonRoot["Id"].asString() + "_" + m_TimeStamp + ".json";
					Write2DbJsonFile(jsonRoot, json_name);
					m_DbJsonFiles[m_Who] = json_name;

					// for show
					std::unique_lock<std::mutex> guardShow(m_MuShow);
					m_DbJsonFile = json_name;
					m_Show = true;
					guardShow.unlock();
					m_CondShow.notify_all();
				}
			}
		TrashFrontBin();
		locker.lock();
	}
}

// read the score, norm2 and feat to the class member variables
// get m_Who through _isSimLgTh()
bool UpdateDB::isFirstTime(std::string& oneFrontJsonFile) {
	Json::Value jsonRoot;
	Json::Value jsonItem;
	ParseJsonFile(oneFrontJsonFile, jsonRoot);
	m_FrontFeatFile = jsonRoot["featFile"].asString();
	m_FrontFaceFile = jsonRoot["faceFile"].asString();
	jsonItem = jsonRoot["timeStamp"];
	m_TimeStamp = jsonItem[1].asString();

	std::ifstream ifs(m_FrontFeatFile, std::ios::binary);
	if (!ifs.is_open()) {
		std::cout << "cannot open file: " << m_FrontFeatFile << std::endl;
		exit(0);
	}
	ifs.read((char *)(&m_FrontScore), sizeof(float));
	ifs.read((char *)(&m_FrontNorm2), sizeof(float));
	ifs.read((char *)(m_Feat), sizeof(float) * FEAT_SIZE);
	ifs.close();
	return !_isSimLgTh();
}

void UpdateDB::CreateBinFiles() {
	CopyFile(m_FrontFaceFile, m_DbFaceFile);
	CopyFile(m_FrontFeatFile, m_DbFeatFile);


	std::unique_lock<std::mutex> guardShow(m_MuShow);
	m_DbJsonFile = m_DbJsonFiles.back();
	m_Show = true;
	guardShow.unlock();
	m_CondShow.notify_all();
}
void UpdateDB::CreateIDFile() {
	m_CNT += 1;    // a new ID will be added
	m_FaceId = Face_ID(std::to_string(m_CNT));
	Json::Value jsonRoot;
	Json::Value jsonItem;
	jsonRoot["Id"] = m_FaceId.asString();
	//jsonRoot["faceFile"] = m_FrontFaceFile;
	jsonRoot["faceFile"] = m_DbFaceDir + "\\ID_" + m_FaceId.asString() + "_" + m_TimeStamp + ".jpg";
	m_DbFaceFile = m_DbFaceDir + "\\ID_" + m_FaceId.asString() + "_" + m_TimeStamp + ".jpg";
	//jsonRoot["featFile"] = m_FrontFeatFile;
	jsonRoot["featFile"] = m_DbFeatDir + "\\FT_" + m_TimeStamp + ".dat";
	m_DbFeatFile = m_DbFeatDir + "\\FT_" + m_TimeStamp + ".dat";
	jsonItem[1] = m_TimeStamp;
	jsonRoot["timeStamp"] = jsonItem;
	m_AppearTimes = 1;
	jsonRoot["appFrqs"] = m_AppearTimes;
	std::string json_name = m_DbJsonDir + "\\ID_" + m_FaceId.asString() + "_" + m_TimeStamp + ".json";
	Write2DbJsonFile(jsonRoot, json_name);
	m_DbJsonFiles.push_back(json_name);
}

void UpdateDB::GenerateDbJsonFileName(std::string& json_name) {
	json_name = m_DbJsonDir + "\\ID_" + m_FaceId.asString() + "_" + m_TimeStamp + ".json";
}

long UpdateDB::initLoadBinData() {
	if (m_DbSize <= 0) {    // not starting 
		return 0;
	}
	long cnt = 0;
	Json::Value jsonRoot;
	for (auto i = 0; i < m_DbSize; i++, cnt = i) {
		ParseJsonFile(m_DbJsonFiles[i], jsonRoot);
		std::string feat_file = jsonRoot["featFile"].asString();
		std::string face_file = jsonRoot["faceFile"].asString();

		std::ifstream ifs(feat_file, std::ios::binary);
		if (!ifs.is_open()) {
			std::cout << "cannot open file: " << feat_file << std::endl;
			exit(0);
		}
		ifs.read((char *)(FEATS + i * (FEAT_SIZE + 2)), 
			sizeof(float) * (FEAT_SIZE + 2));
		ifs.close();

		jsonRoot.clear();
	}
	return cnt;
}

// a float point should point to m_Feat
// return "not the first time"
bool UpdateDB::_isSimLgTh() {
	// the current dB is empty, and the first one comes in
	if (m_FeatNum <= 0) {
		m_Who = 0;
		m_CNT = 0;
		return false;
	}
	float *feat;             // current face
	float *featPt;           //faces in db
	float max_sim = -1.0;
	long max_index = 0;
	bool is_sim_larger = false;

	std::cout << "m_FeatNum = " << m_FeatNum << std::endl;
	//for (auto i = 0; i < m_DbSize; i++) {
	for (auto i = 0; i < m_FeatNum; i++) {
		double acc = 0.0;
		feat = m_Feat;
		featPt = &FEATS[i * (FEAT_SIZE + 2)];
		m_DbScore = *featPt++;
		m_DbNorm2 = *featPt++;
		
		for (int j = 0; j < FEAT_SIZE; j++) {
			acc += (*feat++) * (*featPt++);
		}
		float sim = acc / m_DbNorm2 / m_FrontNorm2;
		if (sim > max_sim) {
			max_sim = sim;
			max_index = i;
		}
	}
	if (max_sim > m_Threshold) {
		//cout << "max_sim = " << max_sim << endl;
		m_Who = max_index;
		return true;   // find the same person
	}
	else {
		m_Who = -1;
		return false;    // cannot find the same person	
	}
}

// m_FeatPos changed
void UpdateDB::_LoadFeats() {
	Json::Value jsonRoot;
	for (long i = 0; i < m_DbJsonFiles.size(); i++) {
		ParseJsonFile(m_DbJsonFiles[i], jsonRoot);
		std::string feat_file = jsonRoot["featFile"].asString();
		std::ifstream ifs(feat_file, std::ios::binary);
		if (!ifs.is_open()) {
			std::cout << "cannot open file " << feat_file << std::endl;
			exit(0);
		}
		ifs.read((char *)(FEATS + i * (FEAT_SIZE + 2)), (FEAT_SIZE + 2) * sizeof(float));
		ifs.close();
		jsonRoot.clear();
		m_FeatPos += (FEAT_SIZE + 2);
	}
}

void UpdateDB::_AppendFeats() {
	FEATS[m_FeatPos] = m_FrontScore;
	FEATS[m_FeatPos + 1] = m_FrontNorm2;
	memcpy((float *)(&FEATS[m_FeatPos + 2]), m_Feat, FEAT_SIZE * sizeof(float));
	m_FeatNum += 1;
	m_FeatPos += (FEAT_SIZE + 2);
}

void UpdateDB::_UpdateFeats() {
	long feat_pos = (FEAT_SIZE + 2) * m_Who;
	FEATS[feat_pos] = m_FrontScore;
	FEATS[feat_pos + 1] = m_FrontNorm2;
	memcpy((float *)(&FEATS[feat_pos + 2]), m_Feat, FEAT_SIZE * sizeof(float));
}

void UpdateDB::Write2DbJsonFile(Json::Value& jsonRoot, std::string& json_name) {
	std::ofstream ofs(json_name);
	if (!ofs.is_open()) {
		std::cout << "fail to open json file: " << json_name << std::endl;
		exit(0);
	}
	ofs << jsonRoot.toStyledString();
	ofs.close();
}

void UpdateDB::TrashFrontBin() {
	DeleteFile(m_FrontFaceFile);
	DeleteFile(m_FrontFeatFile);
	DeleteFile(m_FrontJsonFile);
}



// output the cropped face and its features
void UpdateDB::FaceDsp(MyFace& me, cv::Mat& src_img, cv::Mat& img_gray, float score, float* feats,
	std::vector<MyLandmarks>& landmarks, int best_face, float SimThreshold,
	seeta::FaceIdentification& face_recognizer, bool& isNewJsonFile, int cnt) {

	isNewJsonFile = false;
	//std::cout << "in the chile, isNewJsonFile : " << isNewJsonFile << std::endl;
	// Mr. Hu: initialization
	if (!me.hasFace)
	{
		me.hasFace = true;
		me.color_img = src_img;
		me.gray_img = img_gray;
		me.score = score;
		me.getFeat(feats);
		me = landmarks[best_face].points;
	}
	else {
		float sim = face_recognizer.CalcSimilarity(feats, me.feat);
		std::cout << "cnt = " << cnt << ", sim = " << sim << std::endl;
		// the same ID
		if (sim > SimThreshold) {   // update current me
			if (score > me.score) {
				me.hasFace = true;
				me.color_img = src_img;
				me.gray_img = img_gray;
				me.score = score;
				me.getFeat(feats);
				me = landmarks[best_face].points;
			}
		}
		else {   // a different ID
				 // push me to the files
			std::string time_stamp = getTime();
			std::string face_name;
			std::string feat_name;

			Write2BinFile(me, face_recognizer, time_stamp, face_name, feat_name);
			Write2FrontJsonFile(time_stamp, face_name, feat_name);
			isNewJsonFile = true;

			// complete prduce a new me
			me.hasFace = true;
			me.color_img = src_img;
			me.gray_img = img_gray;
			me.score = score;
			me.getFeat(feats);
			me = landmarks[best_face].points;
		}
	}
}


void UpdateDB::Write2BinFile(MyFace& me, seeta::FaceIdentification& face_recognizer,
	std::string& time_stamp, std::string& face_name, std::string& feat_name) {
	seeta::ImageData img_data(me.color_img.cols, me.color_img.rows, me.color_img.channels());
	img_data.data = me.color_img.data;
	face_name = FRONT_DIR + "\\Faces\\" + "ID_" + time_stamp + ".jpg";
	// face image: 256 x 256
	face_recognizer.CropWithoutExtractFeature(face_name, img_data, me.landmarks);
	me.hasFace = false;
	//CROP = true;

	me.norm2 = face_recognizer.CalcNorm2(me.feat);
	feat_name = FRONT_DIR + "\\BinDescript\\" + "FT_" + time_stamp + ".dat";
	std::ofstream ofs(feat_name, std::ios::binary);
	if (!ofs.is_open()) {
		std::cout << "cannot open file for feat writing" << std::endl;
		return;
	}
	// me.score, me.norm2, me.feat[2048]
	ofs.write((char *)&(me.score), sizeof(float));
	ofs.write((char *)&(me.norm2), sizeof(float));
	ofs.write((char *)me.feat, 2048 * sizeof(float));
	ofs.close();
}

// the json file name is stored in m_FrontJsonFile
void UpdateDB::Write2FrontJsonFile(std::string& time_stamp, std::string& face_name,
	std::string& feat_name) {
	Json::Value jsonRoot;
	Json::Value jsonItem;

	jsonRoot["faceFile"] = face_name;
	jsonRoot["featFile"] = feat_name;
	jsonItem[1] = time_stamp;
	jsonRoot["appFrqs"] = jsonItem.size() - 1;
	jsonRoot["timeStamp"] = jsonItem;
	
	//std::string json_name = FRONT_DIR + "\\Json\\" + time_stamp + ".json";
	m_FrontJsonFile = FRONT_DIR + "\\Json\\" + time_stamp + ".json";
	//std::string json_str = swriter.write(jsonRoot);
	std::ofstream ofs(m_FrontJsonFile);
	if (!ofs.is_open()) {
		std::cout << "fail to open json file" << std::endl;
	}
	ofs << jsonRoot.toStyledString();
	ofs.close();
}

bool UpdateDB::isQNotEmpty() {
	return !m_Que.empty();
}

/*
bool UpdateDB::isMainTaskAlive() {
	return !bWorking || !m_Que.empty();
}
*/

int UpdateDB::getQSize() {
	return m_Que.size();
}

bool UpdateDB::isThereDbJsonFiles() {
	return ((m_DbJsonFile != "_NOT_BORN_") && m_Show);
}
void UpdateDB::Show() {

	while (bWorking || isQNotEmpty()/*isThereDbJsonFiles()*/) {
		//locker.unlock();
		std::cout << "I will show " << m_DbJsonFile << std::endl;
		std::unique_lock<std::mutex> guardShow(m_MuShow);
		m_CondShow.wait(guardShow, std::bind(&UpdateDB::isThereDbJsonFiles, this));
	
		m_Show = false;
		Json::Value jsonRoot;
		ParseJsonFile(m_DbJsonFile, jsonRoot);
		guardShow.unlock();

		std::string db_face_file = jsonRoot["faceFile"].asString();
		//long appFrqs = std::stol(jsonRoot["appFrqs"].asString());
		std::string stringFrqs = jsonRoot["appFrqs"].asString();
		jsonRoot.clear();
		
		cv::Mat img = cv::imread(db_face_file);
		std::string text = stringFrqs;
		myPutText(img, text);
		cv::namedWindow("Show", cv::WINDOW_AUTOSIZE);
		cv::imshow("Show", img);
		//std::this_thread::sleep_for(std::chrono::milliseconds(30));

		
		char c = cv::waitKey(30);
		if (c == 27) {
			cv::destroyAllWindows();
			break;
		}
		
	}
	
}