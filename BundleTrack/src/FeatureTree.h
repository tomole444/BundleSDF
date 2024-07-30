
// #include <CGAL/Simple_cartesian.h>
// #include <CGAL/point_generators_3.h>
// #include <CGAL/Orthogonal_k_neighbor_search.h>
// #include <CGAL/Search_traits_3.h>

// #include <cmath>
// #include <vector>
// #include <fstream>



// using K = CGAL::Simple_cartesian<double>;
// using Point_3 = K::Point_3;

// using TreeTraits3 = CGAL::Search_traits_3<K>;
// using Neighbor_search = CGAL::Orthogonal_k_neighbor_search<TreeTraits3>;
// using Tree = Neighbor_search::Tree;


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Dimension.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <boost/iterator/zip_iterator.hpp>
#include <utility>

//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
//typedef CGAL::Cartesian_d<double>  Kernel;
typedef CGAL::Epick_d<CGAL::Dimension_tag<4> > Kernel;
//typedef Kernel::Point_3                                     Point_3;
typedef Kernel::Point_d                                     Point_d;
//typedef boost::tuple<Point_3,int>                           Point_and_int;
typedef boost::tuple<Point_d,int>                           Point_and_int;
//typedef CGAL::Search_traits_3<Kernel>                       Traits_base;
typedef CGAL::Search_traits_d<Kernel, CGAL::Dimension_tag<4>>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
  CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
  Traits_base>                                              Traits;
//typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
typedef CGAL::Orthogonal_k_neighbor_search<Traits_base>          K_neighbor_search;
typedef K_neighbor_search::Tree                             Tree;
typedef K_neighbor_search::Distance                         Distance;


// #include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
// #include <CGAL/Fuzzy_sphere.h>
// #include <CGAL/Kd_tree.h>
// #include <CGAL/Orthogonal_k_neighbor_search.h>
// #include <CGAL/Random.h>
// #include <CGAL/Search_traits_2.h>
// #include <CGAL/Search_traits_adapter.h>
// #include <CGAL/Simple_cartesian.h>
// #include <CGAL/algorithm.h>
// #include <CGAL/point_generators_2.h>
// #include <deque>


// using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
// using Point_3  = Kernel::Point_3;

// using Random_points_iterator = CGAL::Random_points_in_sphere_3<Point_3>;
// using Traits_base            = CGAL::Search_traits_3<Kernel>;
// using Traits = CGAL::Search_traits_adapter<Element, Element::GetPoint, Traits_base>;
// using K_neighbor_search = CGAL::Orthogonal_k_neighbor_search<Traits>;
// using Fuzzy_circle      = CGAL::Fuzzy_sphere<Traits>;
// using Tree              = K_neighbor_search::Tree;
// using Distance          = K_neighbor_search::Distance;
#pragma once

class FeatureTree {
public:
    FeatureTree(); //: tree() {}

    // void insert(const std::vector<Point_3>& points);

    // Eigen::Vector3d rotatePoint(const Eigen::Quaterniond& q, const Eigen::Vector3d& point);

    // Point_3 nearestNeighbor(const Eigen::Vector3d& point);

    void testTree();

//private:
    //Kd_tree tree;
};