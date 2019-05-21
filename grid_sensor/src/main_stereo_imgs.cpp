#include <grid_sensor/grid_sensor.hpp>
#include "grid_sensor/grid_sensor_visualization.hpp"
#include <grid_sensor/util.hpp>
#include <fstream>
#include <iostream>
#include <string> 
#include <sstream>
#include <ctime>
#include <vector>


using namespace std;


class dataset_wrapper
{
public:
      dataset_wrapper()
      {
	  n.param ("kitti/img_root_folder", img_root_folder, img_root_folder);	  
	  n.param ("kitti/raw_img_folder", raw_img_folder, raw_img_folder); 
	  n.param ("kitti/depth_img_folder", depth_img_folder, depth_img_folder);
          n.param ("kitti/label_rvm_folder", label_rvm_folder ,label_rvm_folder);
	  n.param ("kitti/trajectory_file", trajectory_file, trajectory_file);
          n.param("kitti/img_names_file", img_names_file, img_names_file);
          
	  n.param ("kitti/label_root_folder", label_root_folder, label_root_folder);
	  n.param ("kitti/label_bin_folder", label_bin_folder, label_bin_folder);
	  n.param ("kitti/superpixel_bin_folder", superpixel_bin_folder, superpixel_bin_folder);
	  n.param ("kitti/label_img_folder", label_img_folder, label_img_folder);
	  n.param ("kitti/evaluation_img_list", truth_img_list_file, truth_img_list_file);
	  n.param ("kitti/superpixel_img_folder", superpixel_img_folder, superpixel_img_folder);
	  n.param ("kitti/reproj_label_folder", saving_reproj_img_folder, saving_reproj_img_folder);
          n.param("kitti/num_eval_imgs",total_img_ind, total_img_ind);
          
	  n.param ("save_proj_imgs", save_proj_imgs, save_proj_imgs);
	  n.param ("use_crf_optimize", use_crf_optimize, use_crf_optimize);
          n.param ("use_rvm", use_rvm, use_rvm);
    
	  raw_img_folder = img_root_folder + raw_img_folder;
	  depth_img_folder = img_root_folder + depth_img_folder;
	  trajectory_file = img_root_folder + trajectory_file;
	  label_bin_folder = label_root_folder + label_bin_folder;
	  superpixel_bin_folder = label_root_folder + superpixel_bin_folder;
	  label_img_folder = label_root_folder + label_img_folder;
	  truth_img_list_file = label_root_folder + truth_img_list_file;
	  superpixel_img_folder = label_root_folder + superpixel_img_folder;
	  saving_reproj_img_folder = label_root_folder + saving_reproj_img_folder;
          prior_pc_folder = label_root_folder + label_rvm_folder;
	  img_names_file = img_root_folder + img_names_file;
	  
	  grid_sensor = new ca::GridSensor(n);
	  grid_visualizer = new ca::GridVisualizer(nh);

	  img_counter=0;
	  total_img_ind=1000;
          //read_img_names(img_names_file, img_names);
          //total_img_ind = img_names.size();
          //std::cout<<"Read img: # is "<<total_img_ind<<std::endl;
	  /*
	  image_width = 1226;
	  image_height = 370;
          	  calibration_mat<<707.0912, 0, 601.8873, // kitti sequence 5
          			  0, 707.0912, 183.1104,
          		  0,   0,   1;
          */        
 	  image_width = 1241;
 	  image_height = 376;
 	  calibration_mat<<718.856, 0, 607.1928, // kitti sequence 15
 			   0, 718.856, 185.2157,
 			   0,   0,   1;
                  
	  grid_sensor->set_up_calibration(calibration_mat,image_height,image_width);
	  
	  init_trans_to_ground<<1, 0, 0, 0,  
                                0, 0, 1, 0,
                                0,-1, 0, 1,
                                0, 0, 0, 1;
	  if (!read_all_pose(trajectory_file,total_img_ind+1,all_poses))    // poses of each frame
		ROS_ERROR_STREAM("cannot read file "<<trajectory_file);
          else
              ROS_INFO_STREAM("finish reading trajectory "<<trajectory_file);
	  
	  // set up label to color
	  frame_label_prob.resize(image_width*image_height,num_class);
	  grid_sensor->label_to_color_mat = get_label_to_color_matrix();
	  grid_sensor->sky_label = get_sky_label();
	  grid_visualizer->set_label_to_color(grid_sensor->label_to_color_mat,grid_sensor->sky_label);

	  // set up reprojection images and reprojection poses
	  if (save_proj_imgs){
		if (!read_evaluation_img_list(truth_img_list_file, evaluation_image_list))
		      ROS_ERROR_STREAM("cannot read file "<<truth_img_list_file);
		grid_sensor->set_up_reprojections(evaluation_image_list.rows());
    //     	     ROS_ERROR_STREAM("depth_img_name  "<<depth_img_name);
		Eigen::Matrix4f curr_transToWolrd;   // a camera space point multiplied by this, goes to world frame.
		curr_transToWolrd.setIdentity();
                std::cout<<"evalution image list length "<<evaluation_image_list.rows()<<std::endl;
		for (int ind=0;ind<evaluation_image_list.rows();ind++)
		{
		    int img_counter=evaluation_image_list[ind];
		    VectorXf curr_posevec=all_poses.row(img_counter);
		    MatrixXf crf_label_eigen = Eigen::Map<MatrixXf_row>(curr_posevec.data(),3,4);
		    curr_transToWolrd.block(0,0,3,4) = crf_label_eigen;
		    curr_transToWolrd=init_trans_to_ground*curr_transToWolrd;
		    curr_transToWolrd(2,3)=1.0; // HACK set height to constant, otherwise bad for occupancy mapping.
		    grid_sensor->all_WorldToBodys[ind]=curr_transToWolrd.inverse();
		    grid_sensor->all_BodyToWorlds[ind]=curr_transToWolrd;
		}
		grid_sensor->reproj_frame_inds = evaluation_image_list;
	  }
	  if (crf_skip_frames>1)
	    std::cout<<"CRF opti skip frames:  "<<crf_skip_frames<<std::endl;
      }
      ros::NodeHandle n,nh;
      std::string img_root_folder,label_root_folder;      
      std::string raw_img_folder,depth_img_folder, label_bin_folder,trajectory_file,superpixel_bin_folder;
      std::string label_img_folder, truth_img_list_file,superpixel_img_folder;
      std::string saving_reproj_img_folder;
    std::string prior_pc_folder;
    std::string label_rvm_folder;
    std::string img_names_file;
    std::vector<std::string> img_names;
    bool save_proj_imgs,use_crf_optimize;
    bool use_rvm;
      
      ca::GridSensor* grid_sensor;
      ca::GridVisualizer* grid_visualizer;
    
      bool exceed_total=false;
      int crf_skip_frames=1;
      void process_frame()
      {
	    if (exceed_total || img_counter>total_img_ind)
	    {
	      ROS_INFO("Exceed maximum images");
	      exceed_total=true;
	      return;
	    }

	    char frame_index_c[256];
	    sprintf(frame_index_c,"%06d",img_counter);  // format into 6 digit
	    std::string frame_index(frame_index_c);
            std::cout<<"img_counter "<<img_counter<<", total_img_ind"<<total_img_ind<<std::endl;
            //std::string frame_index = img_names[img_counter];

	    std::string img_left_name=raw_img_folder+frame_index+".png";
	    std::string depth_img_name=depth_img_folder+frame_index+".png"; 
	    std::string label_bin_name=label_bin_folder+frame_index+".bin";
	    std::string superpixel_bin_name=superpixel_bin_folder+frame_index+".bin";
	    std::string superpixel_img_name=superpixel_img_folder+frame_index+".png";
	    std::string label_img_name=label_img_folder+frame_index+"_color.png";
            std::string prior_pc_xyz_name=prior_pc_folder+frame_index+".txt";
	    
	    cv::Mat raw_left_img = cv::imread(img_left_name, 1);    //rgb data
	    cv::Mat depth_img = cv::imread(depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);      //CV_16UC1
	    cv::Mat label_img = cv::imread(label_img_name, 1);    //label rgb color image
	    cv::Mat superpixel_img = cv::imread(superpixel_img_name, 1);    //label rgb color image
            
            /*
	    if(raw_left_img.empty() )
		std::cout <<  "read image  "<<frame_index << std::endl ;	      
	    else {
                
                std::cout<<" No raw img "<<img_left_name<<", rgb shape "<< raw_left_img.size()<<"\n";
                img_counter++;
		return;
            }

            if ( depth_img.empty() ) {
                std::cout<<"NO depth img data"<<depth_img_name<<"\n";
                img_counter++;
                return;
                }*/
	    if (use_rvm == false && !read_label_prob_bin(label_bin_name,frame_label_prob))
	    {
                ROS_ERROR_STREAM("cannot read label file "<<label_bin_name);
                exceed_total=true;
                //if (use_crf_optimize)
                return;
	    }

            pcl::PointCloud<pcl::PointXYZ> prior_pc_xyz;
            if (use_rvm) {
                read_rvm_prior(prior_pc_xyz_name, prior_pc_xyz, frame_label_prob);
                std::cout<<"read rvm prior "<<prior_pc_xyz_name <<"\n";
            }

            // a camera space point multiplied by this, goes to world frame.
	    Eigen::Matrix4f curr_transToWolrd;   
	    curr_transToWolrd.setIdentity();
	    VectorXf curr_posevec=all_poses.row(img_counter);
	    MatrixXf crf_label_eigen = Eigen::Map<MatrixXf_row>(curr_posevec.data(),3,4);
	    curr_transToWolrd.block(0,0,3,4) = crf_label_eigen;
	    curr_transToWolrd=init_trans_to_ground*curr_transToWolrd;

            std::cout<<"Update map...\n";
            if (use_crf_optimize) 
                grid_sensor->AddDepthImg(frame_index, raw_left_img, label_img, depth_img,superpixel_img,curr_transToWolrd,frame_label_prob);  // update grid's occupancy value and label probabitliy from neural network prior
            else if (use_rvm)
                grid_sensor->AddDepthImg(raw_left_img, label_img, depth_img,superpixel_img, prior_pc_xyz, curr_transToWolrd,  frame_label_prob);  // update grid's occupancy value and label probabitliy from rvm
            else {
                grid_sensor->BuildOccupancyMap(frame_index, raw_left_img, label_img, depth_img, superpixel_img, curr_transToWolrd, frame_label_prob);
                std::cout<<"Built Occup-map, ";
                grid_sensor->LabelFusion(depth_img, curr_transToWolrd, frame_label_prob);
                std::cout<<"Label Fusion complete\n";
            }

	    if (img_counter%crf_skip_frames==0)  // test CRF every 4 frames
		if (use_crf_optimize)
		    grid_sensor->CRF_optimization(superpixel_bin_name);
	    
	    grid_visualizer->publishObstCloud(true); //use_crf_optimize

	    if (save_proj_imgs) // for evaluation purpose
	    {
	      int img_in_vec = check_element_in_vector(img_counter,evaluation_image_list);
	      if (img_in_vec >= 0)
		  grid_sensor->actual_depth_imgs[img_in_vec]=depth_img.clone();  // set depth
	      if ( (img_in_vec>= 0) )//|| (img_counter %4 ==0))  // re-project every four frames. projecting every frame takes much time.
	      {
		  ROS_INFO_STREAM("reproject images at  "<<img_counter);
		  grid_sensor->reproject_to_images(img_counter);  // project onto all past poses
	      }
	      for (int ind=0;ind<evaluation_image_list.rows();ind++)  // save all past images
	      {
		  int img_counter_cc = evaluation_image_list[ind];
		  if (img_counter_cc <= img_counter)
		  {
		      char frame_index_char[256];
		      sprintf(frame_index_char,"%06d",img_counter_cc);  // format into 6 digit
                      std::cout<<" writing image #"<<frame_index_char<<", cc "<<img_counter_cc<<", ind "<<ind<<", img_in_vec "<<img_in_vec<<std::endl;
		      std::string frame_index_str(frame_index_char);
		      std::string reproj_label_img_bw_path=saving_reproj_img_folder+frame_index_str+"_bw.png";
		      std::string reproj_label_img_color_path=saving_reproj_img_folder+frame_index_str+"_color.png";
		      
		      cv::imwrite( reproj_label_img_color_path, grid_sensor->reproj_label_colors[ind]);
		      cv::imwrite( reproj_label_img_bw_path, grid_sensor->reproj_label_maps[ind]);
                   }
	      }
	    }
	    img_counter++;
      }
      
      int img_counter ;
      int total_img_ind ;
      int image_width;
      int image_height;

      // important. neveral change it manually, math the vector size.
      const int num_class=ca::Vector_Xxf().rows();      

      Eigen::MatrixXf all_poses;
      MatrixXf_row frame_label_prob;
      
      Eigen::Matrix4f init_trans_to_ground;  // initil transformation  // multiply a constant      
      
      Eigen::Matrix3f calibration_mat;
      
      // for evaluation
      VectorXi evaluation_image_list;
      std::vector<cv::Mat> reproject_label_imgs;
      std::vector<cv::Mat> reproject_depth_imgs; // depth buffer for images
      std::vector<cv::Mat> reproj_label_color;
};


int main(int argc, char *argv[]) {
      ros::init(argc, argv, "grid_sensor");
      ros::NodeHandle n;
      
      dataset_wrapper image_wrap;
      
      ros::Rate loop_rate(5);// hz      

      while (ros::ok())
      {
	    image_wrap.process_frame();
	    loop_rate.sleep(); 
      }
      
      ros::spin();
      return 0;
}
