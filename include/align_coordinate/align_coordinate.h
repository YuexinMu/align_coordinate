//
// Created by myx on 2025/3/29.
//

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <queue>

#include "sophus/se2.hpp"
#include "sophus/se3.hpp"

namespace align_coordinate{
#define CUR_CLOUD_CAPACITY 20

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;

using SE3 = Sophus::SE3d;
using SE3f = Sophus::SE3f;

using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Mat4d = Eigen::Matrix4d;
using Mat4f = Eigen::Matrix4f;
using Quatd = Eigen::Quaterniond;
using Quatf = Eigen::Quaternionf;


class AlignCoordinate{
//EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
public:
  AlignCoordinate() = default;
  ~AlignCoordinate();

  bool Init(ros::NodeHandle &nh);

  bool AlignPointCloud(const CloudPtr& anchor_cloud_ptr, SE3 &T);
  void SendTrans(double time);
  void AnchorPointCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);

private:
  std::string robot_name_;
  std::string anchor_robot_name_;
  std::string anchor_point_topic_;
  std::string cur_point_topic_;

  std::string world_frame_;
  std::string connected_frame_;
  std::string odom_topic_;

  double pub_frequency_;
  double init_x_;
  double init_y_;
  double init_z_;
  double sample_duration_;
  double align_fitness_score_th_;
  double align_trans_score_th_;
  bool align_coordinate_already_;


  std::optional<double> anchor_last_receive_time_;
  std::optional<double> cur_last_receive_time_;
  std::optional<double> last_fitness_score_;

  SE3 T_anchor_cur_;
  std::deque<CloudPtr> cur_point_queue_;

  nav_msgs::Odometry odometry_;
  ros::Publisher pub_odom_;

  ros::Subscriber sub_cur_cloud_, sub_anchor_cloud_;


  void CurPointCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);

  /// utils
  CloudPtr VoxelCloud(CloudPtr cloud, float voxel_size = 0.1);
  template <typename T>
  void SetPoseStamp(SE3 pose, T &out);
  bool InsertCloud(CloudPtr &cloud);

};

}  // namespace align_coordinate
