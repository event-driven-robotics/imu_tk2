#include <iostream>

#include "imu_tk/io_utils.h"
#include "imu_tk/calibration.h"
#include "imu_tk/filters.h"
#include "imu_tk/integration.h"
#include "imu_tk/visualization.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace fs = boost::filesystem; 
namespace po = boost::program_options;

using namespace std;
using namespace imu_tk;
using namespace Eigen;

int main(int argc, char** argv)
{

  vector< TriadData > acc_data, gyro_data;
  
  string acc_file, gyr_file, suffix;
  double g_mag, init_interval_duration_, gyr_dt_, nominal_1g_norm_, init_acc_bias_, init_gyr_scale_, alpha_;
  int min_num_intervals_, interval_n_samples_, max_num_iterations_; 
  bool acc_use_means_, optimize_gyro_bias_, minimizeAccBiases_, minimizeGyrBiases_, verbose_output_;

  po::options_description po("Options for imu_tk2");
  po.add_options()
      ("help", "print options")
      ("acc_file", po::value<string>(& acc_file)->default_value("acc"), "file containing acc readings")
      ("gyr_file", po::value<string>(& gyr_file)->default_value("gyr"), "file containing gyro readings")
     
      // double params 
      ("g_mag, g", po::value<double>(& g_mag)->default_value(9.805622), "magntute of local gravity")
      ("init_interval_duration", po::value<double>(&init_interval_duration_)->default_value(50.0), "duration of the initial interval")
      ("gyr_dt", po::value<double>(& gyr_dt_)->default_value(-1), "gyro rate, -1 automatic")
      ("nominal_1g_norm", po::value<double>(& nominal_1g_norm_)->default_value(16384.0), "Datasheet value for 1g acc")
      ("alpha", po::value<double>(& alpha_)->default_value(0.75), "weight for bias minimesation objective. Set to 1 if not 0.5 < alpha < 1")

      // int params
      ("min_num_intervals", po::value<int>(& min_num_intervals_)->default_value(9), "Minimum numbr of intervals to perform calibration")
      ("interval_n_samples", po::value<int>(& interval_n_samples_)->default_value(300), "Minimum number of samples for static intervals")
      ("max_iter", po::value<int>(& max_num_iterations_)->default_value(100), "Maximum number of iterations for Ceres Solver")
      
      // bool params
      ("acc_use_means", po::value<bool>(& acc_use_means_)->default_value(false), "Use means for acc")
      ("opt_gyr_b", po::value<bool>(& optimize_gyro_bias_)->default_value(false), "if false calculates the gyro biases bases on the initial static interval")
      ("min_acc_b", po::value<bool>(& minimizeAccBiases_)->default_value(true), "Perform multiobjective opt to minimize the biases")
      ("min_gyr_b", po::value<bool>(& minimizeGyrBiases_)->default_value(true), "Perform multiobjective opt to minimize the biases")
      ("verbose", po::value<bool>(& verbose_output_)->default_value(true), "Print a lot of stuff")
  
      // Initial calibration guesses
      ("init_acc_bias", po::value<double>(& init_acc_bias_)->default_value(1), "Initial guess of the accelerometer bias")
      ("init_gyr_scale", po::value<double>(& init_gyr_scale_)->default_value((250 * M_PI / (2.0 * 180.0 * 16384.0))), "initial guess for gyro scale - from datasheet")
      ("suffix", po::value<string>(& suffix)->default_value("params"), "sufix for calibration results files")
      ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, po), vm);
  po::notify(vm);
 
  if (vm.count("help")) 
  {
      cout << "Usage: options_description [options]\n";
      cout << po << std::endl;
      return 0;
  }

  // Handle file paths  
  boost::replace_all(acc_file, "~", getenv ("HOME"));
  boost::replace_all(gyr_file, "~", getenv ("HOME"));

  cout<<"Importing IMU data from the Matlab matrix file : "<< acc_file << endl;  
  importAsciiData( acc_file.c_str(), acc_data, imu_tk::TIMESTAMP_UNIT_SEC );
  cout<<"Importing IMU data from the Matlab matrix file : "<< gyr_file << endl;  
  importAsciiData( gyr_file.c_str(), gyro_data, imu_tk::TIMESTAMP_UNIT_SEC  );

  if(acc_data.size()==0)
  {
    cout << "\nNo data in the files... exiting" << endl;
    return 1;
  }

  
  MultiPosCalibration mp_calib;
  
  //double params 
  mp_calib.setGravityMagnitude(g_mag);
  mp_calib.setInitStaticIntervalDuration(init_interval_duration_);
  mp_calib.setGyroDataPeriod(gyr_dt_);
  mp_calib.set1g(nominal_1g_norm_);
  mp_calib.setAlpha(alpha_);

  // int params
  mp_calib.setMinNumIntervals(min_num_intervals_);
  mp_calib.setIntarvalsNumSamples(interval_n_samples_);
  mp_calib.setMaxIte(max_num_iterations_);

  // bool params
  mp_calib.enableAccUseMeans(acc_use_means_);
  mp_calib.enableGyroBiasOptimization(optimize_gyro_bias_);
  mp_calib.enableAccBiasMin(minimizeAccBiases_);
  mp_calib.enableGyrBiasMin(minimizeGyrBiases_);
  mp_calib.enableVerboseOutput(verbose_output_);
 
  // string params
  mp_calib.setPlotFile(acc_file + "_" + suffix + ".png");
  
  //init_acc_calib.setBias( Vector3d(32768, 32768, 32768) );
  //init_gyro_calib.setScale( Vector3d(1.0/6258.0, 1.0/6258.0, 1.0/6258.0) );

  // Set initial guess for calibration 
  CalibratedTriad init_acc_calib, init_gyro_calib;
  init_acc_calib.setBias( Vector3d(init_acc_bias_, init_acc_bias_, init_acc_bias_) );
  init_gyro_calib.setScale( Vector3d(init_gyr_scale_, init_gyr_scale_, init_gyr_scale_) );
  mp_calib.setInitAccCalibration( init_acc_calib );
  mp_calib.setInitGyroCalibration( init_gyro_calib );  
  
  // Perform the calibration
  mp_calib.calibrateAccGyro(acc_data, gyro_data );
  
  // Save results 
  string acc_params_name = acc_file + "." + suffix;
  string gyr_params_name = gyr_file + "." + suffix;

  cout << "file names:\n" << acc_params_name << "\n" << gyr_params_name << "\n\n";
  mp_calib.getAccCalib().save(acc_params_name);
  mp_calib.getGyroCalib().save(gyr_params_name);
  
  return 0;
}
