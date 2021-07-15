/* 
 * imu_tk - Inertial Measurement Unit Toolkit
 * 
 *  Copyright (c) 2014, Alberto Pretto <pretto@diag.uniroma1.it>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Extended by: Leandro de Souza Rosa <leandro.desouzarosa@iit.it>
 *  Istituto Italiano di tecnologia, Genova, Italy
 *  Date 01, July, 2021
 */

#include "imu_tk/calibration.h"
#include "imu_tk/filters.h"
#include "imu_tk/integration.h"
#include "imu_tk/visualization.h"

#include <limits>
#include <iostream>
#include "ceres/ceres.h"

using namespace imu_tk;
using namespace Eigen;
using namespace std;

template <typename _T1> struct MultiPosAccResidual
{
  MultiPosAccResidual( const _T1 &g_mag, const _T1 &alpha, const Eigen::Matrix< _T1, 3 , 1> &sample ) :
  g_mag_(g_mag),
  alpha_(alpha),
  sample_(sample){}
  
  template <typename _T2>
    bool operator() ( const _T2* const params, _T2* residuals ) const
  {
    Eigen::Matrix< _T2, 3 , 1> raw_samp( _T2(sample_(0)), _T2(sample_(1)), _T2(sample_(2)) );
    /* Assume body frame same as accelerometer frame, 
     * so bottom left params in the misalignment matris are set to zero */
    CalibratedTriad_<_T2> calib_triad( params[0], params[1], params[2], 
                                     _T2(0), _T2(0), _T2(0),
                                     params[3], params[4], params[5], 
                                     params[6], params[7], params[8] );
    
    Eigen::Matrix< _T2, 3 , 1> calib_samp = calib_triad.unbiasNormalize( raw_samp );
    residuals[0] = _T2(alpha_)*(_T2 ( g_mag_ ) - calib_samp.norm());
    
    return true;
  }
  
  static ceres::CostFunction* Create ( const _T1 &g_mag, const _T1 &alpha, const Eigen::Matrix< _T1, 3 , 1> &sample )
  {
    return ( new ceres::AutoDiffCostFunction< MultiPosAccResidual, 1, 9 > (
               new MultiPosAccResidual<_T1>( g_mag, alpha, sample ) ) );
  }
  
  const _T1 g_mag_;
  const _T1 alpha_;
  const Eigen::Matrix< _T1, 3 , 1> sample_;
};

template <typename _T1> struct BiasesMinimizeResidual
{
  BiasesMinimizeResidual(const _T1 &g_mag, const _T1 &alpha):
  g_mag_(g_mag),
  alpha_(alpha){}

  template <typename _T2>
    bool operator() ( const _T2* const params, _T2* residuals ) const
  {
    Eigen::Matrix< _T2, 3 , 1> zeros( _T2(0), _T2(0), _T2(0) );
    CalibratedTriad_<_T2> calib_triad( params[0], params[1], params[2], 
                                     _T2(0), _T2(0), _T2(0),
                                     params[3], params[4], params[5], 
                                     params[6], params[7], params[8] );
    
    Eigen::Matrix< _T2, 3 , 1> calib_biases = calib_triad.unbiasNormalize( zeros );

    residuals[0] = _T2(1-alpha_)*calib_biases.norm();
    return true;
  }
  
  static ceres::CostFunction* Create ( const _T1 &g_mag, const _T1 &alpha )
  {
    return ( new ceres::AutoDiffCostFunction< BiasesMinimizeResidual, 1, 9 > (
               new BiasesMinimizeResidual<_T1>( g_mag, alpha ) ) );
  }
  
  const _T1 alpha_;
  const _T1 g_mag_;
};

//-------------------------------------- Gyro residual ----------------------------------------------
template <typename _T1> struct MultiPosGyroResidual
{
  MultiPosGyroResidual( const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos0, 
                        const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos1,
                        const std::vector< TriadData_<_T1> > &gyro_samples, 
                        const DataInterval &gyro_interval_pos01, 
                        _T1 dt, bool optimize_bias) :

  g_versor_pos0_(g_versor_pos0), 
  g_versor_pos1_(g_versor_pos1),
  gyro_samples_(gyro_samples),
  interval_pos01_(gyro_interval_pos01),
  dt_(dt), optimize_bias_(optimize_bias){}
  
  template <typename _T2>
    bool operator() ( const _T2* const params, _T2* residuals ) const
  {
    CalibratedTriad_<_T2> calib_triad( params[0], params[1], params[2], 
                                      params[3], params[4], params[5], 
                                      params[6], params[7], params[8],
                                      optimize_bias_?params[9]:_T2(0), 
                                      optimize_bias_?params[10]:_T2(0), 
                                      optimize_bias_?params[11]:_T2(0) );

    std::vector< TriadData_<_T2> > calib_gyro_samples;
    calib_gyro_samples.reserve( interval_pos01_.end_idx - interval_pos01_.start_idx + 1 );
    
    for( int i = interval_pos01_.start_idx; i <= interval_pos01_.end_idx; i++ )
      calib_gyro_samples.push_back( TriadData_<_T2>( calib_triad.unbiasNormalize( gyro_samples_[i] ) ) );
    
    Eigen::Matrix< _T2, 3 , 3> rot_mat;
    integrateGyroInterval( calib_gyro_samples, rot_mat, _T2(dt_) );
    
    Eigen::Matrix< _T2, 3 , 1> diff = rot_mat.transpose()*g_versor_pos0_.template cast<_T2>() -
                                      g_versor_pos1_.template cast<_T2>();
    
    residuals[0] = diff(0);
    residuals[1] = diff(1);
    residuals[2] = diff(2);
    
    return true;
  }
  
  static ceres::CostFunction* Create ( const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos0, 
                                       const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos1,
                                       const std::vector< TriadData_<_T1> > &gyro_samples, 
                                       const DataInterval &gyro_interval_pos01, 
                                       _T1 dt, bool optimize_bias )
  {
    if( optimize_bias )
      return ( new ceres::AutoDiffCostFunction< MultiPosGyroResidual, 3, 12 > (
                new MultiPosGyroResidual( g_versor_pos0, g_versor_pos1, gyro_samples, 
                                          gyro_interval_pos01, dt, optimize_bias ) ) );
    else
      return ( new ceres::AutoDiffCostFunction< MultiPosGyroResidual, 3, 9 > (
                new MultiPosGyroResidual( g_versor_pos0, g_versor_pos1, gyro_samples, 
                                          gyro_interval_pos01, dt, optimize_bias ) ) );
  }
  
  const Eigen::Matrix< _T1, 3 , 1> g_versor_pos0_, g_versor_pos1_;
  const std::vector< TriadData_<_T1> > gyro_samples_;
  const DataInterval interval_pos01_;
  const _T1 dt_;
  const bool optimize_bias_;
};

template <typename _T1> struct GyroBiasMinimizeResidual
{
  GyroBiasMinimizeResidual( const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos0, 
                        const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos1,
                        const std::vector< TriadData_<_T1> > &gyro_samples, 
                        const DataInterval &gyro_interval_pos01, 
                        _T1 dt, bool optimize_bias) :

  g_versor_pos0_(g_versor_pos0), 
  g_versor_pos1_(g_versor_pos1),
  gyro_samples_(gyro_samples),
  interval_pos01_(gyro_interval_pos01),
  dt_(dt), optimize_bias_(optimize_bias){}
  
  template <typename _T2>
    bool operator() ( const _T2* const params, _T2* residuals ) const
  {
    CalibratedTriad_<_T2> calib_triad( params[0], params[1], params[2], 
                                      params[3], params[4], params[5], 
                                      params[6], params[7], params[8],
                                      optimize_bias_?params[9]:_T2(0), 
                                      optimize_bias_?params[10]:_T2(0), 
                                      optimize_bias_?params[11]:_T2(0) );

    std::vector< TriadData_<_T2> > calib_gyro_samples;
    calib_gyro_samples.reserve( interval_pos01_.end_idx - interval_pos01_.start_idx + 1 );
   
    for( int i = interval_pos01_.start_idx; i <= interval_pos01_.end_idx; i++ )
    {
        TriadData_<_T2> zeros = TriadData_<_T2>(_T2(gyro_samples_[i].timestamp()), _T2(0), _T2(0), _T2(0));
        calib_gyro_samples.push_back( TriadData_<_T2>( calib_triad.unbiasNormalize( zeros ) ) );
    }


    Eigen::Matrix< _T2, 3 , 3> rot_mat;
    integrateGyroInterval( calib_gyro_samples, rot_mat, _T2(dt_) );
    
    Eigen::Matrix< _T2, 3 , 1> diff = rot_mat.transpose()*g_versor_pos0_.template cast<_T2>(); 
    
    residuals[0] = diff(0);
    residuals[1] = diff(1);
    residuals[2] = diff(2);
    
    return true;
  }
  
  static ceres::CostFunction* Create ( const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos0, 
                                       const Eigen::Matrix< _T1, 3 , 1> &g_versor_pos1,
                                       const std::vector< TriadData_<_T1> > &gyro_samples, 
                                       const DataInterval &gyro_interval_pos01, 
                                       _T1 dt, bool optimize_bias )
  {
    return ( new ceres::AutoDiffCostFunction< GyroBiasMinimizeResidual, 3, 12 > (
        new GyroBiasMinimizeResidual( g_versor_pos0, g_versor_pos1, gyro_samples, 
                                  gyro_interval_pos01, dt, optimize_bias ) ) );
  }
  
  const Eigen::Matrix< _T1, 3 , 1> g_versor_pos0_, g_versor_pos1_;
  const std::vector< TriadData_<_T1> > gyro_samples_;
  const DataInterval interval_pos01_;
  const _T1 dt_;
  const bool optimize_bias_;
};

//-------------------------------- Optimize routines ----------------------------------

template <typename _T>
  MultiPosCalibration_<_T>::MultiPosCalibration_() :
  g_mag_(9.805622),
  min_num_intervals_(9),
  init_interval_duration_(_T(50.0)),
  interval_n_samples_(300),
  acc_use_means_(false),
  gyro_dt_(-1.0),
  optimize_gyro_bias_(true),
  max_num_iterations_(100),
  acc_bias_bound_multiplier_(2),
  gyr_bias_bound_multiplier_(2),
  nominal_1g_norm_(16384.0),
  minimizeAccBiases_(true), 
  minimizeGyrBiases_(true),
  verbose_output_(false){}

template <typename _T>
  bool MultiPosCalibration_<_T>::calibrateAcc ( const std::vector< TriadData_<_T> >& acc_samples )
{
  cout<<"Accelerometers calibration: calibrating..."<<endl;
  cout << "gravity mag: " << g_mag_ << endl;
  min_cost_static_intervals_.clear();
  calib_acc_samples_.clear();
  calib_gyro_samples_.clear();
  
  int n_samps = acc_samples.size();
  
  DataInterval init_static_interval = DataInterval::initialInterval( acc_samples, init_interval_duration_ );
  Eigen::Matrix<_T, 3, 1> acc_variance = dataVariance( acc_samples, init_static_interval );
  _T norm_th = acc_variance.norm(); 

  //calculate a value for the biases upper and lower bound based 
  //on the biases in the initial static interval
  Eigen::Matrix<_T, 3, 1> acc_initial_avg = dataMean(acc_samples, init_static_interval);
  std::cout << "\n\nacc initial acc avg norm: " << acc_initial_avg.norm();
  _T initial_average_bias = abs(acc_initial_avg.norm() - nominal_1g_norm_); 
  std::cout << "\nacc initial bias avg: " << initial_average_bias;
  acc_bias_bound = acc_bias_bound_multiplier_*initial_average_bias;
  std::cout << "\nacc bias bound: " << acc_bias_bound << "\n\n";

  _T min_cost = std::numeric_limits< _T >::max();

  int min_cost_th = -1;
  std::vector< double > min_cost_calib_params;
  
  ceres::Solver::Summary summary;
  
  for (int th_mult = 2; th_mult <= 10; th_mult++)
  {
    std::vector< imu_tk::DataInterval > static_intervals;
    std::vector< imu_tk::TriadData_<_T> > static_samples;
    std::vector< double > acc_calib_params(9);
    
    acc_calib_params[0] = init_acc_calib_.misYZ();
    acc_calib_params[1] = init_acc_calib_.misZY();
    acc_calib_params[2] = init_acc_calib_.misZX();
    
    acc_calib_params[3] = init_acc_calib_.scaleX();
    acc_calib_params[4] = init_acc_calib_.scaleY();
    acc_calib_params[5] = init_acc_calib_.scaleZ();
    
    acc_calib_params[6] = init_acc_calib_.biasX();
    acc_calib_params[7] = init_acc_calib_.biasY();
    acc_calib_params[8] = init_acc_calib_.biasZ();
    
    std::vector< DataInterval > extracted_intervals;
    staticIntervalsDetector ( acc_samples, th_mult*norm_th, static_intervals );
    extractIntervalsSamples ( acc_samples, static_intervals, 
                              static_samples, extracted_intervals,
                              interval_n_samples_, acc_use_means_ );
    
    if(verbose_output_)
      cout<<"Accelerometers calibration: extracted "<<extracted_intervals.size()
          <<" intervals using threshold multiplier "<<th_mult<<" -> ";
    
    // TODO Perform here a quality test
    if( extracted_intervals.size() < min_num_intervals_)
    {
      if( verbose_output_) cout<<"Not enough intervals, calibration is not possible"<<endl;
      continue;
    }
    
    if( verbose_output_) cout<<"Trying calibrate... "<<endl;
    
    ceres::Problem problem;
    
    if(!minimizeAccBiases_){
        cout << "setting alpha = 1 since we are not minimizing biases" << endl;
        alpha_ = 1;
    }
    
    for( int i = 0; i < static_samples.size(); i++)
    {
      // Add acc calibration cost function
      ceres::CostFunction* cost_function_calib = MultiPosAccResidual<_T>::Create ( g_mag_, alpha_, static_samples[i].data() );
      problem.AddResidualBlock ( cost_function_calib, NULL /* squared loss */, acc_calib_params.data() );
      
      // Add bias reduction cost function
      if(minimizeAccBiases_)
      {  
        ceres::CostFunction* cost_function_biases = BiasesMinimizeResidual<_T>::Create( g_mag_, alpha_);
        problem.AddResidualBlock ( cost_function_biases, NULL /* squared loss */, acc_calib_params.data() );

        //add lower and upped bound for biases
        //biases x, y, z are idxs 6, 7, 8 of acc_calib_params.data() array
        problem.SetParameterLowerBound(acc_calib_params.data(), 6, -acc_bias_bound); 
        problem.SetParameterUpperBound(acc_calib_params.data(), 6,  acc_bias_bound);

        problem.SetParameterLowerBound(acc_calib_params.data(), 7, -acc_bias_bound);  
        problem.SetParameterUpperBound(acc_calib_params.data(), 7,  acc_bias_bound); 
    
        problem.SetParameterLowerBound(acc_calib_params.data(), 8, -acc_bias_bound);  
        problem.SetParameterUpperBound(acc_calib_params.data(), 8,  acc_bias_bound); 
      }
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose_output_;
    options.max_num_iterations = max_num_iterations_;

    ceres::Solve ( options, &problem, &summary );
   
    std::cout << "\n\nSolver performed that many iterations: " << summary.iterations.back().iteration << "\n\n"; 
    if(summary.final_cost < min_cost && summary.iterations.back().iteration < max_num_iterations_)
    {
      min_cost = summary.final_cost;
      min_cost_th = th_mult;
      min_cost_static_intervals_ = static_intervals;
      min_cost_calib_params = acc_calib_params;
    }
    cout<<"residual "<<summary.final_cost<<endl;
  }
  
  if( min_cost_th < 0 )
  {
    if(verbose_output_) 
      cout<<"Accelerometers calibration: Can't obtain any calibratin with the current dataset"<<endl;
    return false;
  }

  acc_calib_ = CalibratedTriad_<_T>( min_cost_calib_params[0],
                                     min_cost_calib_params[1],
                                     min_cost_calib_params[2],
                                     0,0,0,
                                     min_cost_calib_params[3],
                                     min_cost_calib_params[4],
                                     min_cost_calib_params[5],
                                     min_cost_calib_params[6],
                                     min_cost_calib_params[7],
                                     min_cost_calib_params[8] );
  
  calib_acc_samples_.reserve(n_samps);
  
  // Calibrate the input accelerometer data with the obtained calibration
  for( int i = 0; i < n_samps; i++)
    calib_acc_samples_.push_back( acc_calib_.unbiasNormalize( acc_samples[i]) );
  
  if(verbose_output_) 
  {
    Plot plot;
    plot.plotIntervals( calib_acc_samples_, min_cost_static_intervals_);
    plot.save(plotFile_); 
    cout<<summary.FullReport()<<endl;
    cout<<"Accelerometers calibration: Better calibration obtained using threshold multiplier "<<min_cost_th
        <<" with residual "<<min_cost<<endl
        <<acc_calib_<<endl
        <<"Accelerometers calibration: inverse scale factors:"<<endl
        <<1.0/acc_calib_.scaleX()<<endl
        <<1.0/acc_calib_.scaleY()<<endl
        <<1.0/acc_calib_.scaleZ()<<endl;
        
    //waitForKey();
  }
  
  return true;
    
}

template <typename _T> 
  bool MultiPosCalibration_<_T>::calibrateAccGyro ( const vector< TriadData_<_T> >& acc_samples, 
                                                   const vector< TriadData_<_T> >& gyro_samples )
{
  if( !calibrateAcc( acc_samples ) )
    return false;
  
  cout<<"Gyroscopes calibration: calibrating..."<<endl;
  
  std::vector< TriadData_<_T> > static_acc_means;
  std::vector< DataInterval > extracted_intervals;
  extractIntervalsSamples ( calib_acc_samples_, min_cost_static_intervals_, 
                            static_acc_means, extracted_intervals,
                            interval_n_samples_, true );
  
  int n_static_pos = static_acc_means.size(), n_samps = gyro_samples.size();
  
  // Compute the gyroscopes biases in the (static) initialization interval
  DataInterval init_static_interval = DataInterval::initialInterval( gyro_samples, init_interval_duration_ );
  Eigen::Matrix<_T, 3, 1> gyro_bias = dataMean( gyro_samples, init_static_interval );
  
  gyro_calib_ = CalibratedTriad_<_T>(0, 0, 0, 0, 0, 0, 
                                    1.0, 1.0, 1.0, 
                                    gyro_bias(0), gyro_bias(1), gyro_bias(2) );
  
  // Calculate bound for gyro biases
  std::cout << "\n\ngyr initial bias avg: \n" << gyro_bias;
  gyr_bias_bound = gyr_bias_bound_multiplier_*gyro_bias.array().abs();
  std::cout << "\ngyr bias bound: \n" << gyr_bias_bound << "\n\n";

  // calib_gyro_samples_ already cleared in calibrateAcc()
  calib_gyro_samples_.reserve(n_samps);
  // Remove the bias
  for( int i = 0; i < n_samps; i++ )
    calib_gyro_samples_.push_back(gyro_calib_.unbias(gyro_samples[i]));
  
  std::vector< double > gyro_calib_params(12);

  gyro_calib_params[0] = init_gyro_calib_.misYZ();
  gyro_calib_params[1] = init_gyro_calib_.misZY();
  gyro_calib_params[2] = init_gyro_calib_.misZX();
  gyro_calib_params[3] = init_gyro_calib_.misXZ();
  gyro_calib_params[4] = init_gyro_calib_.misXY();
  gyro_calib_params[5] = init_gyro_calib_.misYX();
  
  gyro_calib_params[6] = init_gyro_calib_.scaleX();
  gyro_calib_params[7] = init_gyro_calib_.scaleY();
  gyro_calib_params[8] = init_gyro_calib_.scaleZ();
  
  // Bias has been estimated and removed in the initialization period
  gyro_calib_params[9]  = 0.0; //gyro_bias[0]; //0.0;
  gyro_calib_params[10] = 0.0; //gyro_bias[1]; //0.0;
  gyro_calib_params[11] = 0.0; //gyro_bias[2]; //0.0;
  
  ceres::Problem problem;
      
  for( int i = 0, t_idx = 0; i < n_static_pos - 1; i++ )
  {
    Eigen::Matrix<_T, 3, 1> g_versor_pos0 = static_acc_means[i].data(),
                            g_versor_pos1 = static_acc_means[i + 1].data();
                               
    g_versor_pos0 /= g_versor_pos0.norm();                           
    g_versor_pos1 /= g_versor_pos1.norm();
    
    int gyro_idx0 = -1, gyro_idx1 = -1;
    _T ts0 = calib_acc_samples_[extracted_intervals[i].end_idx].timestamp(), 
       ts1 = calib_acc_samples_[extracted_intervals[i + 1].start_idx].timestamp();
     
    // Assume monotone signal time
    for( ; t_idx < n_samps; t_idx++ )
    {
      if( gyro_idx0 < 0 )
      {
        if( calib_gyro_samples_[t_idx].timestamp() >= ts0 )
          gyro_idx0 = t_idx;
      }
      else
      {
        if( calib_gyro_samples_[t_idx].timestamp() >= ts1 )
        {
          gyro_idx1 = t_idx - 1;
          break;
        }
      }
    }
    
//     cout<<"from "<<calib_gyro_samples_[gyro_idx0].timestamp()<<" to "
//         <<calib_gyro_samples_[gyro_idx1].timestamp()
//         <<" v0 : "<< g_versor_pos0(0)<<" "<< g_versor_pos0(1)<<" "<< g_versor_pos0(2)
//         <<" v1 : "<< g_versor_pos1(0)<<" "<< g_versor_pos1(1)<<" "<< g_versor_pos1(2)<<endl;
    
    DataInterval gyro_interval(gyro_idx0, gyro_idx1);
   
    //  Add acc calibration cost function    
    ceres::CostFunction* cost_function_calib =  MultiPosGyroResidual<_T>::Create ( g_versor_pos0, g_versor_pos1, calib_gyro_samples_, gyro_interval, gyro_dt_, optimize_gyro_bias_ );
    problem.AddResidualBlock ( cost_function_calib, NULL /* squared loss */, gyro_calib_params.data() );

    //  Add bias reduction cost function
    if(minimizeGyrBiases_)
    {
        ceres::CostFunction* cost_function_biases =  GyroBiasMinimizeResidual<_T>::Create ( g_versor_pos0, g_versor_pos1, calib_gyro_samples_, gyro_interval, gyro_dt_, optimize_gyro_bias_ );
        problem.AddResidualBlock ( cost_function_biases, NULL /* squared loss */, gyro_calib_params.data() );

        //add lower and upped bound for biases
        //biases x, y, z are idxs 9, 10, 11 of gyro_calib_params.data() array
        problem.SetParameterLowerBound(gyro_calib_params.data(),  9, -gyr_bias_bound[0]); 
        problem.SetParameterUpperBound(gyro_calib_params.data(),  9,  gyr_bias_bound[0]);
                                                                                   
        problem.SetParameterLowerBound(gyro_calib_params.data(), 10, -gyr_bias_bound[1]);  
        problem.SetParameterUpperBound(gyro_calib_params.data(), 10,  gyr_bias_bound[1]); 
                                                                                   
        problem.SetParameterLowerBound(gyro_calib_params.data(), 11, -gyr_bias_bound[2]);  
        problem.SetParameterUpperBound(gyro_calib_params.data(), 11,  gyr_bias_bound[2]); 
    }
  }
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = verbose_output_;
  options.max_num_iterations = max_num_iterations_;
  ceres::Solver::Summary summary;

  ceres::Solve ( options, &problem, &summary );
  gyro_calib_ = CalibratedTriad_<_T>( gyro_calib_params[0],
                                     gyro_calib_params[1],
                                     gyro_calib_params[2],
                                     gyro_calib_params[3],
                                     gyro_calib_params[4],
                                     gyro_calib_params[5],
                                     gyro_calib_params[6],
                                     gyro_calib_params[7],
                                     gyro_calib_params[8],
                                     gyro_bias(0) + gyro_calib_params[9],
                                     gyro_bias(1) + gyro_calib_params[10],
                                     gyro_bias(2) + gyro_calib_params[11]);                            
  cout << "\n\n gyro bias opt params: " << gyro_calib_params[9] << ", " << gyro_calib_params[10] << ", " << gyro_calib_params[11] << "\n\n"; 
  // Calibrate the input gyroscopes data with the obtained calibration
  for( int i = 0; i < n_samps; i++)
    calib_gyro_samples_.push_back( gyro_calib_.unbiasNormalize( gyro_samples[i]) );
  
  if(verbose_output_) 
  {
    cout<<summary.FullReport()<<endl;
    cout<<"Gyroscopes calibration: residual "<<summary.final_cost<<endl
        <<gyro_calib_<<endl
        <<"Gyroscopes calibration: inverse scale factors:"<<endl
        <<1.0/gyro_calib_.scaleX()<<endl
        <<1.0/gyro_calib_.scaleY()<<endl
        <<1.0/gyro_calib_.scaleZ()<<endl;
  }
  
  return true;
}

template class MultiPosCalibration_<double>;
template class MultiPosCalibration_<float>;
