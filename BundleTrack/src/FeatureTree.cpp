#include "FeatureTree.h"



// void FeatureTree::insert(const std::vector<Point_d>& points, const std::vector<int>& indices) {
//     tree.insert(boost::make_zip_iterator(boost::make_tuple( points.begin(),indices.begin())),
//                 boost::make_zip_iterator(boost::make_tuple( points.end(),indices.end())));
// }
// void FeatureTree::insert(const Eigen::Quaternion<double>& quat, int index){
    
//     tree.insert(boost::make_tuple( FeatureTree::quaternionToPoint_D(quat), index));
// }

void FeatureTree::insert(std::shared_ptr<Frame>& frame){
    Eigen::Quaternion<double> quat = FeatureTree::getQuaternionFromFrame(frame);
    insert(quat, frame);
}


void FeatureTree::insert(const Eigen::Quaternion<double>& quat, std::shared_ptr<Frame>& frame){
    tree.insert(boost::make_tuple( FeatureTree::quaternionToPoint_D(quat), frame));
}

K_neighbor_search FeatureTree::nearestNeighbor(std::shared_ptr<Frame>& frame, unsigned int K) {
    Eigen::Quaternion<double> quat = FeatureTree::getQuaternionFromFrame(frame);
    Point_d query = FeatureTree::quaternionToPoint_D(quat);
    return K_neighbor_search(tree, query, K) ;
}

K_neighbor_search FeatureTree::nearestNeighbor(const Eigen::Quaternion<double>& quat, unsigned int K) {
    Point_d query = FeatureTree::quaternionToPoint_D(quat);
    return K_neighbor_search(tree, query, K) ;
}

Point_d FeatureTree::quaternionToPoint_D(const Eigen::Quaternion<double>& quat){
    return Point_d (quat.w(),quat.x(),quat.y(),quat.z());
}
Eigen::Quaternion<double> FeatureTree::getQuaternionFromFrame (std::shared_ptr<Frame>& frame) {
    Eigen::Matrix4f pose = frame->_pose_in_model;
    Eigen::Quaternion<double> quat(pose.block(0,0,3,3));
    return quat;
}
void FeatureTree::testTree(){
   

    // const unsigned int K = 5;
    // std::vector<Point_d> points;
    // std::vector<int>     indices;
    // points.push_back(Point_d(3.0, 6.0, 7.0, 2.0));
    // points.push_back(Point_d(17.0, 15.0, 13.0, 2.0));
    // points.push_back(Point_d(13.0, 15.0, 6.0, 2.0));
    // points.push_back(Point_d(6.0, 12.0, 8.0, 2.0));
    // points.push_back(Point_d(9.0, 1.0, 2.0, 2.0));
    // points.push_back(Point_d(2.0, 7.0, 6.0, 2.0));
    // points.push_back(Point_d(1.0, 2.0, 3.0, 2.0));
    // indices.push_back(0);
    // indices.push_back(1);
    // indices.push_back(2);
    // indices.push_back(3);
    // indices.push_back(4);
    // indices.push_back(5);
    // indices.push_back(6);

    // // Insert number_of_data_points in the tree
    // Tree tree(boost::make_zip_iterator(boost::make_tuple( points.begin(),indices.begin())),
    //             boost::make_zip_iterator(boost::make_tuple( points.end(),indices.end())));
    // // Tree tree(points.begin(), points.end());
    // // search K nearest neighbors
    // Point_d query(1.0, 1.0, 1.0, 1.0);
    // std::cout << "Query: "<< query << std::endl; 
    // Distance tr_dist;
    // K_neighbor_search search(tree, query, K);
    // for(K_neighbor_search::iterator it = search.begin(); it != search.end(); it++){

    //     // std::cout << it->first << "  "<< std::sqrt(it->second) << std::endl;
    //     std::cout << " d(q, nearest neighbor)=  "
    //             << tr_dist.inverse_of_transformed_distance(it->second) << " "
    //             << boost::get<0>(it->first)<< " " << boost::get<1>(it->first) << std::endl;
    // }

    
    exit(0);
}

void FeatureTree::testClass(){
   

    const unsigned int K = 5;
    std::shared_ptr<Frame> frame0 = std::make_shared<Frame>();
    frame0->_id = 0;
    std::shared_ptr<Frame> frame1 = std::make_shared<Frame>();
    frame1->_id = 1;
    std::shared_ptr<Frame> frame2 = std::make_shared<Frame>();
    frame2->_id = 2;
    std::shared_ptr<Frame> frame3 = std::make_shared<Frame>();
    frame3->_id = 3;
    std::shared_ptr<Frame> frame4 = std::make_shared<Frame>();
    frame4->_id = 4;
    std::shared_ptr<Frame> frame5 = std::make_shared<Frame>();
    frame5->_id = 5;
    std::shared_ptr<Frame> frame6 = std::make_shared<Frame>();
    frame6->_id = 6;

    insert(Eigen::Quaternion<double> (3.0, 6.0, 7.0, 2.0), frame0);
    insert(Eigen::Quaternion<double> (7.0, 15.0, 13.0, 2.0), frame1);
    insert(Eigen::Quaternion<double> (3.0, 15.0, 6.0, 2.0), frame2);
    insert(Eigen::Quaternion<double> (6.0, 12.0, 8.0, 2.0), frame3);
    insert(Eigen::Quaternion<double> (9.0, 1.0, 2.0, 2.0), frame4);
    insert(Eigen::Quaternion<double> (2.0, 7.0, 6.0, 2.0), frame5);
    insert(Eigen::Quaternion<double> (1.0, 2.0, 3.0, 2.0), frame6);

    Eigen::Quaternion<double> query (1.0,1.0,1.0,1.0);
    Distance tr_dist;
    K_neighbor_search search = nearestNeighbor(query, K);
    for(K_neighbor_search::iterator it = search.begin(); it != search.end(); it++){

        // std::cout << it->first << "  "<< std::sqrt(it->second) << std::endl;
        std::cout << " d(q, nearest neighbor)=  "
                << tr_dist.inverse_of_transformed_distance(it->second) << " index: "
                << boost::get<0>(it->first)<< " " << boost::get<1>(it->first)->_id << std::endl;
    }

    
    exit(0);
}