#pragma once


#include <CGAL/Epick_d.h>
#include <CGAL/Dimension.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <boost/iterator/zip_iterator.hpp>
#include <utility>

#include <Eigen/Geometry> 
#include "Frame.h"

class Frame;

using Kernel = CGAL::Epick_d<CGAL::Dimension_tag<4> >;
using Point_d = Kernel::Point_d;
using Point_and_int =  boost::tuple<Point_d,int>;
using Point_and_Frame =  boost::tuple<Point_d, std::shared_ptr<Frame>>;
using Traits_base = CGAL::Search_traits_d<Kernel, CGAL::Dimension_tag<4>> ;
using Traits = CGAL::Search_traits_adapter<Point_and_Frame, CGAL::Nth_of_tuple_property_map<0, Point_and_Frame>, Traits_base>;
using K_neighbor_search = CGAL::Orthogonal_k_neighbor_search<Traits> ;
using Tree = K_neighbor_search::Tree;
using Distance = K_neighbor_search::Distance;


class FeatureTree {
public:
    FeatureTree(): tree(){}

    // void insert(const std::vector<Point_d>& points, const std::vector<int>& indices);
    // void insert(const Eigen::Quaternion<double>& quat, int index);
    void insert(std::shared_ptr<Frame>& frame);
    void insert(const Eigen::Quaternion<float>& quat, std::shared_ptr<Frame>& frame);

    K_neighbor_search nearestNeighbor(const Eigen::Quaternion<float>& quat, unsigned int K);
    K_neighbor_search nearestNeighbor(std::shared_ptr<Frame>& frame, unsigned int K);

    static Point_d quaternionToPoint_D (const Eigen::Quaternion<float>& quat);
    static Eigen::Quaternion<float> getQuaternionFromFrame (std::shared_ptr<Frame>& frame);

    void testTree();

    void testClass();

private:
    Tree tree;
};