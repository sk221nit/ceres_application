#include "ceres_app.h"
#include "ceres_base.h"

using namespace ceresapp;
using namespace ceres;

void BuildProblem(BundleAdjustment *bundleAdjustment, Problem *problem)
{
    const int point_block_size = bundleAdjustment->point_block_size();
    const int camera_block_size = bundleAdjustment->camera_block_size();
    double *points = bundleAdjustment->mutable_points();
    double *cameras = bundleAdjustment->mutable_cameras();

    // Observations is 2*num_observations long array observations =
    // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
    // and y positions of the observation.
    const double *observations = bundleAdjustment->observations();
    for (int i = 0; i < bundleAdjustment->num_observations(); ++i)
    {
        CostFunction *cost_function;
        // Each Residual block takes a point and a camera as input and
        // outputs a 2 dimensional residual.
        cost_function = SnavelyReprojectionErrorWithQuaternions::Create(
            observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        LossFunction *loss_function = new ceres::CauchyLoss(1.0);

        // Each observation correponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera =
            cameras + camera_block_size * bundleAdjustment->camera_index()[i];
        double *point = points + point_block_size * bundleAdjustment->point_index()[i];
        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }

    LocalParameterization *camera_parameterization =
        new ProductParameterization(new QuaternionParameterization(),
                                    new IdentityParameterization(6));
    for (int i = 0; i < bundleAdjustment->num_cameras(); ++i)
    {
        problem->SetParameterization(cameras + camera_block_size * i,
                                     camera_parameterization);
    }
}

int main(int argc, char **argv)
{
    std::cout << "RUN Ceres Application " << argv[1] << std::endl;
    google::InitGoogleLogging(argv[0]);
    std::string fileName = argv[1];

    BundleAdjustment bundleAdjustment(fileName, true);
    bundleAdjustment.WriteToFile("ceresfile.csv");

    Problem problem;
    BuildProblem(&bundleAdjustment, &problem);

    Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summery;
    ceres::Solve(options, &problem, &summery);
    std::cout << "\n"
              << summery.FullReport() << "\n"
              << std::endl;

    bundleAdjustment.WriteToFile("ceresoutfile.csv");

    // double initial_x = 5.0;
    // double x = initial_x;
    // ceres::Problem problem;

    // ceres::CostFunction* cost_function =  new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    // problem.AddResidualBlock(cost_function, nullptr, &x);

    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_progress_to_stdout = true;
    // ceres::Solver::Summary summery;
    // ceres::Solve(options, &problem, &summery);
    // std::cout<<summery.BriefReport()<<"\n";
    // std::cout<<"x: "<<initial_x<<" --->  "<<x<<"\n";
    return 0;
}
