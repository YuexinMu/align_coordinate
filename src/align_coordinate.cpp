//
// Created by myx on 2025/3/29.
//

#include "align_coordinate/align_coordinate.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <tf2_ros/transform_broadcaster.h>

namespace align_coordinate{

AlignCoordinate::~AlignCoordinate(){
  cur_point_queue_.clear();
}

bool AlignCoordinate::Init(ros::NodeHandle &nh) {
  nh.param<std::string>("anchor_robot_name", anchor_robot_name_, "sct0");
  nh.param<std::string>("robot_name", robot_name_, "sct1");
  nh.param<std::string>("anchor_point_topic", anchor_point_topic_, "no_anchor_point_topic");
  nh.param<std::string>("cur_point_topic", cur_point_topic_, "no_cur_point_topic");
  nh.param<std::string>("align_odom_topic", odom_topic_, "map2world_odom");

  nh.param<double>("init_x", init_x_, 0.0);
  nh.param<double>("init_y", init_y_, 0.0);
  nh.param<double>("init_z", init_z_, 0.0);

  nh.param<std::string>("/world_frame", world_frame_, "world");
  nh.param<std::string>("connected_frame", connected_frame_, "map");

//  nh.param<double>("pub_frequency", pub_frequency_, 10.0);
  nh.param<double>("sample_duration", sample_duration_, 1.0);

  nh.param<double>("align_fitness_score_th", align_fitness_score_th_, 1.0);
  nh.param<double>("align_trans_score_th", align_trans_score_th_, 5.0);

  ROS_INFO_STREAM("align_fitness_score_th: " << align_fitness_score_th_);
  ROS_INFO_STREAM("align_trans_score_th: " << align_trans_score_th_);

  pub_odom_ = nh.advertise<nav_msgs::Odometry>(odom_topic_, 10);

  odometry_.header.frame_id = world_frame_;
  odometry_.child_frame_id = connected_frame_;

  align_coordinate_already_ = false;
  T_anchor_cur_ = SE3();

  sub_anchor_cloud_ = nh.subscribe<sensor_msgs::PointCloud2>(anchor_point_topic_,
                                                             20, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                                                             { AnchorPointCallBack(msg); });

  sub_cur_cloud_ = nh.subscribe<sensor_msgs::PointCloud2>(cur_point_topic_,
                                                         20, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                                                         { CurPointCallBack(msg); });
  ROS_INFO("AlignCoordinate::AlignPointCloud Init complete.");
  return true;
}

bool AlignCoordinate::AlignPointCloud(const CloudPtr& anchor_cloud_ptr, SE3 &T) {
  if(anchor_cloud_ptr == nullptr || cur_point_queue_.empty()){
    ROS_INFO_STREAM("anchor_cloud_ptr is nullptr.");
    return false;
  }
  CloudPtr cur_frame{new PointCloudType};

  for(const auto& item : cur_point_queue_){
    if(item == nullptr){
      ROS_INFO_STREAM("item is nullptr.");
      return false;
    }
    *cur_frame += *item;
  }

  Mat4f T_anchor_cur_ori = T_anchor_cur_.matrix().cast<float>();;
  CloudPtr cloud_trans(new PointCloudType);

  /// 不同分辨率下的匹配
  pcl::NormalDistributionsTransform<PointType, PointType> ndt;
  ndt.setTransformationEpsilon(0.05);
  ndt.setStepSize(0.5);
  ndt.setMaximumIterations(40);
  CloudPtr output(new PointCloudType);
  std::vector<double> res{10.0, 5.0, 4.0, 3.0};
  for (auto& r : res) {
    ndt.setResolution(r);
    ndt.setStepSize(r*0.1);
    auto rough_map1 = VoxelCloud(anchor_cloud_ptr, r * 0.1);
    auto rough_map2 = VoxelCloud(cur_frame, r * 0.1);
    ndt.setInputTarget(rough_map1);
    ndt.setInputSource(rough_map2);

    ndt.align(*output, T_anchor_cur_ori);
    T_anchor_cur_ori = ndt.getFinalTransformation();
  }

  Mat4d T_ori_d = T_anchor_cur_ori.cast<double>();
  Quatd q(T_ori_d.block<3, 3>(0, 0));
  q.normalize();
  Vec3d t = T_ori_d.block<3, 1>(0, 3);
  SE3 origin_T = SE3(q, t);

  if(!ndt.hasConverged()){
    ROS_INFO_STREAM("Robot : " << robot_name_
                               << " cloud align(ndt) not converged with robot: "
                               << anchor_robot_name_);
    return false;
  }

  double fitness_score = ndt.getFitnessScore();
  // fitness score: sum[1-n](d^2) / n
  if(fitness_score > align_fitness_score_th_){
    ROS_INFO_STREAM("The fitness score too large (" << fitness_score
                               << ") between robot: " << robot_name_ << " and "
                               << anchor_robot_name_);
    return false;
  }
  double trans_score = ndt.getTransformationProbability();
  if(trans_score < align_trans_score_th_){
    ROS_INFO_STREAM("The transformation score too small (" << trans_score
                                                    << ") between robot: " << robot_name_ << " and "
                                                    << anchor_robot_name_);
    return false;
  }

  if(last_fitness_score_.has_value()){
    if(fitness_score < last_fitness_score_){
      last_fitness_score_ = fitness_score;
      T = origin_T;
      ROS_INFO("Update align value. New fitness score: %d.", fitness_score);
      return true;
    } else{
      return false;
    }
  }
  last_fitness_score_ = fitness_score;
  T = origin_T;
  return true;
}

void AlignCoordinate::SendTrans(double time) {
  odometry_.header.stamp = ros::Time().fromSec(time);
  SetPoseStamp(T_anchor_cur_, odometry_.pose);
  pub_odom_.publish(odometry_);

  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transform;

  transform.header.stamp = odometry_.header.stamp;
  transform.header.frame_id = world_frame_;
  transform.child_frame_id = connected_frame_;

  transform.transform.translation.x = odometry_.pose.pose.position.x;
  transform.transform.translation.y = odometry_.pose.pose.position.y;
  transform.transform.translation.z = odometry_.pose.pose.position.z;
  transform.transform.rotation = odometry_.pose.pose.orientation;

  br.sendTransform(transform);
}

void AlignCoordinate::AnchorPointCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  ROS_INFO_ONCE("Received point cloud of anchor robot: %s.", anchor_robot_name_.c_str());
  if(robot_name_ == anchor_robot_name_){
    return;
  }
  if(anchor_last_receive_time_.has_value()){
    double cur_receive_time = msg->header.stamp.toSec();
    if(cur_receive_time - anchor_last_receive_time_.value() > sample_duration_){
      CloudPtr anchor_cloud{new PointCloudType};
      pcl::fromROSMsg(*msg, *anchor_cloud);

      if(AlignPointCloud(anchor_cloud, T_anchor_cur_)){
        ROS_INFO_ONCE("Align cloud between robot: %s and %s Successful.",
                      robot_name_.c_str(), anchor_robot_name_.c_str());
        SendTrans(msg->header.stamp.toSec());
        align_coordinate_already_ = true;
      }
      anchor_last_receive_time_ = cur_receive_time;
    }
  } else{
    anchor_last_receive_time_ = msg->header.stamp.toSec();
  }
}

void AlignCoordinate::CurPointCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  ROS_INFO_ONCE("Received point cloud of cur robot: %s.", robot_name_.c_str());
  if(robot_name_ == anchor_robot_name_){
    T_anchor_cur_ = SE3();
    SendTrans(msg->header.stamp.toSec());
    // Anchor robot is itself.
    return;
  } else{
    if(align_coordinate_already_){
      SendTrans(msg->header.stamp.toSec());
    } else{
      T_anchor_cur_.translation().x() = init_x_;
      T_anchor_cur_.translation().y() = init_y_;
      T_anchor_cur_.translation().z() = init_z_;
      SendTrans(msg->header.stamp.toSec());
    }
  }
  if(cur_last_receive_time_.has_value()){
    double cur_receive_time = msg->header.stamp.toSec();
    if(cur_receive_time - cur_last_receive_time_.value() > sample_duration_){
      CloudPtr cur_cloud{new PointCloudType};
      pcl::fromROSMsg(*msg, *cur_cloud);
      InsertCloud(cur_cloud);

      cur_last_receive_time_ = msg->header.stamp.toSec();
    }
  } else{
    cur_last_receive_time_ = msg->header.stamp.toSec();
    return;
  }
}

CloudPtr AlignCoordinate::VoxelCloud(CloudPtr cloud, float voxel_size) {
  pcl::VoxelGrid<PointType> voxel;
  voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel.setInputCloud(cloud);

  CloudPtr output(new PointCloudType);
  voxel.filter(*output);

  return output;
}

template <typename T>
void AlignCoordinate::SetPoseStamp(SE3 pose, T &out) {
  out.pose.position.x = pose.translation().x();
  out.pose.position.y = pose.translation().y();
  out.pose.position.z = pose.translation().z();
  out.pose.orientation.x = pose.so3().unit_quaternion().x();
  out.pose.orientation.y = pose.so3().unit_quaternion().y();
  out.pose.orientation.z = pose.so3().unit_quaternion().z();
  out.pose.orientation.w = pose.so3().unit_quaternion().w();
}

bool AlignCoordinate::InsertCloud(CloudPtr &cloud) {
  if(cloud == nullptr){
    ROS_WARN_STREAM("[AlignCoordinate::InsertCloud] Insert Cloud is null.");
    return false;
  }
  if (cur_point_queue_.size() >= CUR_CLOUD_CAPACITY) {
    cur_point_queue_.erase(cur_point_queue_.begin());
  }
  cur_point_queue_.push_back(cloud);
  return true;
}

}


int main(int argc, char **argv) {
  ros::init(argc, argv, "align_coordinate");
  ros::NodeHandle nh;

  auto align = std::make_shared<align_coordinate::AlignCoordinate>();
  align->Init(nh);

  ros::spin();
//  ros::Rate rate(50);
//  while (ros::ok()) {
//
//    ros::spinOnce();
//    rate.sleep();
//  }

  return 0;
}

