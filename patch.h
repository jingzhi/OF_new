
// Class implements step (3.) in Algorithm 1 of the paper:
// It finds the displacement of one patch from reference/template image to the closest-matching patch in target image via gradient descent.


#ifndef PAT_HEADER
#define PAT_HEADER

                                              //#include <opencv2/core/core.hpp> // needed for verbosity >= 3, DISVISUAL
                                              //#include <opencv2/highgui/highgui.hpp> // needed for verbosity >= 3, DISVISUAL
                                              //#include <opencv2/imgproc/imgproc.hpp> // needed for verbosity >= 3, DISVISUAL

#include "oflow.h" // For camera intrinsic and opt. parameter struct

namespace OFC
{

  
typedef struct
{
  bool hasconverged;
  bool hasoptstarted;
  bool hasconverged_2nd;
  bool hasoptstarted_2nd;
          
  // reference/template patch 
  Eigen::Matrix<float, Eigen::Dynamic, 1> pdiff; // image error to reference image
  Eigen::Matrix<float, Eigen::Dynamic, 1> pweight; // absolute error image
  Eigen::Matrix<float, Eigen::Dynamic, 1> pdiff_2nd; // image error to reference image
  Eigen::Matrix<float, Eigen::Dynamic, 1> pweight_2nd; // absolute error image

  #if (SELECTMODE==1) // Optical Flow
  Eigen::Matrix<float, 2, 2> Hes; // Hessian for optimization
  Eigen::Vector2f p_in, p_iter, delta_p; // point position, displacement to starting position, iteration update
  Eigen::Vector2f p_in_2nd, p_iter_2nd, delta_p_2nd; // point position, displacement to starting position, iteration update
  #else // Depth from Stereo
  Eigen::Matrix<float, 1, 1> Hes; // Hessian for optimization
  Eigen::Matrix<float, 1, 1> p_in, p_iter, delta_p; // point position, displacement to starting position, iteration update
  #endif
  
  // start positions, current point position, patch norm
  Eigen::Matrix<float,1,1> normtmp;
  Eigen::Vector2f pt_iter; 
  Eigen::Vector2f pt_st;
  Eigen::Vector2f pt_iter_2nd; 
  Eigen::Matrix<float, Eigen::Dynamic, 1> isInlier;
    
  float delta_p_sqnorm = 1e-10;
  float delta_p_sqnorm_init = 1e-10; 
  float mares = 1e20; // mares: Mean Absolute RESidual
  float mares_old = 1e20;
  int cnt=0;
  bool invalid=false;
  float delta_p_sqnorm_2nd = 1e-10;
  float delta_p_sqnorm_init_2nd = 1e-10; 
  float mares_2nd = 1e20; // mares: Mean Absolute RESidual
  float mares_old_2nd = 1e20;
  int cnt_2nd=0;
  bool invalid_2nd=true;
  
} patchstate;


  
class PatClass
{
  
public:
  PatClass(const camparam* cpt_in,
            const camparam* cpo_in,
            const optparam* op_in,
            const int patchid_in);

  ~PatClass();


  //****Public Methods Declaration of this class*****//

  void InitializePatch(Eigen::Map<const Eigen::MatrixXf> * im_ao_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dy_in, const Eigen::Vector2f pt_ref_in);
  void SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * im_bo_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dy_in);

  #if (SELECTMODE==1) // Optical Flow
  void OptimizeIter(const Eigen::Vector2f p_in_arg, const bool untilconv);
  #else  // Depth from Stereo
  void OptimizeIter(const Eigen::Matrix<float, 1, 1> p_in_arg, const bool untilconv);
  #endif  

  inline const bool isConverged() const { return pc->hasconverged; }
  inline const bool isConverged_2nd() const { return pc->hasconverged_2nd; }
  inline const bool hasOptStarted() const { return pc->hasoptstarted; }
  inline const bool hasOptStarted_2nd() const { return pc->hasoptstarted_2nd; }
  inline const Eigen::Vector2f GetPointPos() const { return pc->pt_iter; }  // get current iteration patch position (in this frame's opposite camera for OF, Depth)
  inline const bool IsValid() const { return (!pc->invalid) ; }
  inline const bool IsValid_2nd() const { return (!pc->invalid_2nd) ; }
  inline const float GetpWeight() const {return (float) pc->pweight.array().mean()+sqrt((pc->pweight.array() - pc->pweight.array().mean()).square().sum()/(pc->pweight.array().size()-1)); }   
  inline const float GetpWeight_2nd() const {return (float) pc->pweight_2nd.array().mean()+sqrt((pc->pweight_2nd.array() -pc->pweight_2nd.array().mean()).square().sum()/(pc->pweight_2nd.array().size()-1)); } 
  inline const float * GetpWeightPtr() const {return (float*) pc->pweight.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class
  inline const float * GetpWeightPtr_2nd() const {return (float*) pc->pweight_2nd.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class


  #if (SELECTMODE==1) // Optical Flow
  inline const Eigen::Vector2f*            GetParam()    const { return &(pc->p_iter); }   // get current iteration parameters
  inline const Eigen::Vector2f*            GetParam_2nd()    const { return &(pc->p_iter_2nd); }   // get current iteration parameters
  #else // Depth from Stereo
  inline const Eigen::Matrix<float, 1, 1>* GetParam()    const { return &(pc->p_iter); }   // get current iteration parameters
  #endif

  #if (SELECTMODE==1) // Optical Flow
  inline const Eigen::Vector2f*            GetParamStart() const { return &(pc->p_in); }   
  #else // Depth from Stereo
  inline const Eigen::Matrix<float, 1, 1>* GetParamStart() const { return &(pc->p_in); }     
  #endif
  
  //get outlier indicator array pointer
  inline const float * GetpSelectorPtr() const {return (float*) pc->isInlier.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class

private:
  //*******Private Method declaration********//
  
  #if (SELECTMODE==1) // Optical Flow
  void OptimizeStart(const Eigen::Vector2f p_in_arg);
  void OptimizeStart_2nd(const Eigen::Vector2f p_in_arg);
  #else // Depth from Stereo
  void OptimizeStart(const Eigen::Matrix<float, 1, 1> p_in_arg);  
  #endif
  
  void OptimizeComputeErrImg();
  void OptimizeComputeErrImg_2nd();
  void paramtopt();
  void paramtopt_2nd();
  void ResetPatch();
  void ResetPatch_2nd();
  void ComputeHessian();
  void CreateStatusStruct(patchstate * psin);
  void LossComputeErrorImage(Eigen::Matrix<float, Eigen::Dynamic, 1>* patdest,  Eigen::Matrix<float, Eigen::Dynamic, 1>* wdest, const Eigen::Matrix<float, Eigen::Dynamic, 1>* patin,  const Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmpin);
  
  // Extract patch on integer position, and gradients, No Bilinear interpolation
  void getPatchStaticNNGrad    (const float* img, const float* img_dx, const float* img_dy,  const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in,  Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmp_dx_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_dy_in);
  // Extract patch on float position with bilinear interpolation, no gradients.  
  void getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in,  Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e);
  
  //*****Private Data********//

  Eigen::Vector2f pt_ref; // reference point location, pixel coordinates of mid-point of the patch
  Eigen::Matrix<float, Eigen::Dynamic, 1> tmp;
  Eigen::Matrix<float, Eigen::Dynamic, 1> tmp_2nd;
  Eigen::Matrix<float, Eigen::Dynamic, 1> dxx_tmp; // x derivative, doubles as steepest descent image for OF, Depth, SF
  Eigen::Matrix<float, Eigen::Dynamic, 1> dyy_tmp; // y derivative, doubles as steepest descent image for OF, SF
  //scratch container
  Eigen::Map<const Eigen::MatrixXf> * im_ao, * im_ao_dx, * im_ao_dy;
  Eigen::Map<const Eigen::MatrixXf> * im_bo, * im_bo_dx, * im_bo_dy;
  
  const camparam* cpt;
  const camparam* cpo;
  const optparam* op; 
  const int patchid;
  
  patchstate * pc = nullptr; // current patch state
    
};


}

#endif /* PAT_HEADER */


