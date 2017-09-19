#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  n_x_ = 5;
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  time_us_ = 0.0;

  n_aug_ = 7;

  n_sig_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  weights_ = VectorXd::Zero(n_sig_);

  lambda_ = 3 - n_aug_;

  NIS_LIDAR_ = 0.0;

  NIS_RADAR_ = 0.0;

  R_radar_ = MatrixXd::Zero(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_laser_ = MatrixXd::Zero(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_; 
                
  time_step_ = 0;

  // set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;

  for (int i = 1; i < n_sig_; i++)
  {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_)
  {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {

      cout << "Sensor type: RADAR" << endl;
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);

      double px = rho * cos(phi);
      double py = rho * sin(phi);
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      x_ << px, py, v, 0, 0;
      cout << "x_ is done!" << endl;
      // P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
      //       0, std_radr_ * std_radr_, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, std_radphi_, 0,
      //       0, 0, 0, 0, std_radphi_;
      // P_ << 1, 0, 0, 0, 0,
      //       0, 1, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, 1, 0,
      //       0, 0, 0, 0, 1;
      // cout << "P_ is done!" << endl;
   
      // cout << "R is done!" << endl;
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {

      cout << "Sensor type: LASER" << endl;
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);

      x_ << px, py, 0, 0, 0;

      // P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
      //       0, std_laspy_ * std_laspy_, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, 1, 0,
      //       0, 0, 0, 0, 1;
      // P_ << 1, 0, 0, 0, 0,
      //       0, 1, 0, 0, 0,
      //       0, 0, 1, 0, 0,
      //       0, 0, 0, 1, 0,
      //       0, 0, 0, 0, 1;
     
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    cout << "Done Initializing!" << endl;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // while (dt > 0.1)
  // {
  //   const double delta_t = 0.05;
  //   Prediction(delta_t);
  //   dt -= delta_t;
  // }
  /* double increment = 0.1; */

  /* if (dt >= increment * 2) */
  /* { */
  /*   while (dt >= increment * 2) */
  /*   { */
  /*     Prediction(increment); */
  /*     dt -= increment; */
  /*   } */
  /* } */

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    /* cout << "Updating RADAR" << endl; */
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    /* cout << "Updating LASER" << endl; */
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  cout << "Time step: " << time_step_ << endl;
  cout << "Delta t: " << delta_t << endl;
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);

  /* Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_); */

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // Sigma point prediction
  for (int i = 0; i < n_sig_; i++)
  {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  VectorXd x_pred = VectorXd::Zero(n_x_);

  MatrixXd P_pred = MatrixXd::Zero(n_x_, n_x_);

  //predct state mean
  // x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  {
    x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
  }

  cout << "x_: " << x_ << endl;
  cout << "P_: " << P_ << endl;
  /* cout << "weights: " << weights_ << endl; */
  /* cout << "Xsig_pred_: " << Xsig_pred_ << endl; */
  //predicted state covariance matrix
  // P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;

    while (x_diff(3)> M_PI)
    {
      // cout << "Normalizing angle..." << endl; 
      x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI)
    {
      // cout << "Normalizing angle..." << endl; 
      x_diff(3)+=2.*M_PI;
    }

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose() ;

  }
  x_ = x_pred;
  P_ = P_pred;
  cout << "Angle is OK!" << endl; 
  time_step_++;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;

  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sig_);

  for (int i = 0; i < n_sig_; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  VectorXd z_pred = VectorXd::Zero(n_z);
  /* z_pred = Zsig * weights_; */
  for (int i = 0; i < n_sig_; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < n_sig_; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    /* while (z_diff(3)> M_PI) z_diff(3)-=2.*M_PI; */
    /* while (z_diff(3)<-M_PI) z_diff(3)+=2.*M_PI; */
    //angle normalization

    while (z_diff(1)> M_PI)
    {
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI)
    {
      z_diff(1)+=2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_laser_;

  /* VectorXd z = meas_package.raw_measurements_; */
  VectorXd z = VectorXd(n_z);
  double temp_px = meas_package.raw_measurements_(0);
  double temp_py = meas_package.raw_measurements_(1);
  z << temp_px,
       temp_py;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // for (int i = 0; i < n_sig_; i++)
  // {
  //   VectorXd z_diff = Zsig.col(i) - z_pred;
  //   VectorXd x_diff = Xsig_pred_.col(i) - x_;
  //   Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  // }

  for (int i = 0; i < n_sig_; i++)
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI)
    {
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI)
    {
      z_diff(1)+=2.*M_PI;
    }
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI)
    {
        x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI)
    {
        x_diff(3)+=2.*M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI)
  {
    z_diff(1)-=2.*M_PI;
  }
  while (z_diff(1)<-M_PI)
  {
    z_diff(1)+=2.*M_PI;
  }

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_LIDAR_ = z_diff.transpose() * S.inverse() * z_diff;

  cout << "Update Lidar complete!" << endl; 
}
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z = 3;

  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sig_);

  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  VectorXd z_pred = VectorXd::Zero(n_z);
  /* z_pred = Zsig * weights_; */
  for (int i = 0; i < n_sig_; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  for (int i = 0; i < n_sig_; i++)
  {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI)
    {
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI)
    {
      z_diff(1)+=2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix

  S = S + R_radar_;

  /* VectorXd z = meas_package.raw_measurements_; */
  VectorXd z = VectorXd(n_z);
  double temp_rho = meas_package.raw_measurements_(0);
  double temp_phi = meas_package.raw_measurements_(1);
  double temp_rhodot = meas_package.raw_measurements_(2);
  z << temp_rho,
       temp_phi,
       temp_rhodot;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);


  //calculate cross correlation matrix
  for (int i = 0; i < n_sig_; i++)
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI)
    {
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI)
    {
      z_diff(1)+=2.*M_PI;
    }
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI)
    {
        x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI)
    {
        x_diff(3)+=2.*M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI)
  {
    z_diff(1)-=2.*M_PI;
  }
  while (z_diff(1)<-M_PI)
  {
    z_diff(1)+=2.*M_PI;
  }

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_RADAR_ = z_diff.transpose() * S.inverse() * z_diff;

  cout << "Update Radar complete!" << endl; 
}
