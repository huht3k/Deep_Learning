#include "face_registration.h"


int main(int argc, char* argv[]) {
	std::cout << "Hello World!" << std::endl;

	UpdateDB myDB(DB_DIR, FRONT_JSON_DIR, DB_FT_DIR, DB_FACE_DIR);
	//myDB.VisionFaces();
	//myDB.Show();
	 
	auto mainTask = std::bind(&UpdateDB::DataMaking, &myDB);
	auto frontTask = std::bind(&UpdateDB::VisionFaces, &myDB);
	auto showTask = std::bind(&UpdateDB::Show, &myDB);

	std::thread mainThread(mainTask);
	std::thread VisionFacesThread(frontTask);
	std::thread showAppearence(showTask);
	mainThread.join();
	VisionFacesThread.join();
	showAppearence.join();

	std::cout << "Q size: " << myDB.getQSize() << std::endl;
	


	return 0;
}