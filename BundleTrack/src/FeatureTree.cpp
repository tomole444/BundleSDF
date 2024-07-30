#include "FeatureTree.h"

// void FeatureTree::insert(const std::vector<Point_3>& points) {
//     tree.insert(points.begin(), points.end());
// }

// Eigen::Vector3d FeatureTree::rotatePoint(const Eigen::Quaterniond& q, const Eigen::Vector3d& point) {
//     return q * point;
// }

// Point_3 FeatureTree::nearestNeighbor(const Eigen::Vector3d& point) {
//     Point_3 query(point[0], point[1], point[2]);
//     return *tree.closest_point(query);
// }

FeatureTree::FeatureTree(){

}

void FeatureTree::testTree(){
   

    const unsigned int K = 5;
    std::vector<Point_d> points;
    std::vector<int>     indices;
    points.push_back(Point_d(3.0, 6.0, 7.0, 2.0));
    points.push_back(Point_d(17.0, 15.0, 13.0, 2.0));
    points.push_back(Point_d(13.0, 15.0, 6.0, 2.0));
    points.push_back(Point_d(6.0, 12.0, 8.0, 2.0));
    points.push_back(Point_d(9.0, 1.0, 2.0, 2.0));
    points.push_back(Point_d(2.0, 7.0, 6.0, 2.0));
    points.push_back(Point_d(1.0, 2.0, 3.0, 2.0));
    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);
    indices.push_back(3);
    indices.push_back(4);
    indices.push_back(5);
    indices.push_back(6);

    // Insert number_of_data_points in the tree
    // Tree tree(boost::make_zip_iterator(boost::make_tuple( points.begin(),indices.begin())),
    //             boost::make_zip_iterator(boost::make_tuple( points.end(),indices.end())));
    Tree tree(points.begin(), points.end());
    // search K nearest neighbors
    Point_d query(1.0, 1.0, 1.0, 1.0);
    std::cout << "Query: "<< query << std::endl; 
    Distance tr_dist;
    K_neighbor_search search(tree, query, K);
    for(K_neighbor_search::iterator it = search.begin(); it != search.end(); it++){

        std::cout << it->first << "  "<< std::sqrt(it->second) << std::endl;
        // std::cout << " d(q, nearest neighbor)=  "
        //         << tr_dist.inverse_of_transformed_distance(it->second) << " "
        //         << boost::get<0>(it->first)<< " " << boost::get<1>(it->first) << std::endl;
    }

    
    exit(0);
}