
<div align="center">

<h1>SegSplat: Semantic-Guided 3D Gaussian Splatting for Sparse View Reconstruction and Segmentation</h1>

<div>
    <a href='' target='_blank'>S Meena Padnekar</a><sup>1</sup>&emsp;
    <a href='https://www.ee.iitm.ac.in/kmitra/' target='_blank'>Kaushik Mitra</a><sup>2</sup>&emsp;
    <a href='https://www.cse.iitm.ac.in/~sdas/' target='_blank'>Sukhendu Das</a><sup>1</sup>&emsp;
   
   
</div>
<div>
    <sup>1</sup>Visualization & Perception Lab, Department of Computer Science and Engineering, Indian Institute of Technology&emsp; 
    <sup>2</sup>Computational Imaging Lab, Department of Electrical Engineering, Indian Institute of Technology&emsp; 

</div>

<div>
    <strong> 2025</strong>
</div>

<div>
    <h4 align="center">
        • <a href="" target='_blank'>[arXiv]</a> •
    </h4>
</div>

<img src="Segmentation.png" /> 
SegSplat performs joint 3D reconstruction and segmentation by leveraging semantic prior from sparse views. Semantic priors enrich the Gaussian representation, enabling geometry refinement, sharper object boundaries, and consistent semantics across views.

</div>

## Abstract
> *Sparse-view 3D reconstruction remains fundamentally ill-posed, where limited observations lead to ambiguity in both geometry and appearance. While 3D Gaussian Splatting offers fast and efficient rendering, existing few-shot variants struggle with texture blurring, structural artifacts, and an absence of semantic awareness.
We introduce SemanticSplat, a unified framework that leverages the mutual information between geometry and semantics to enhance photorealistic reconstruction and scene understanding from sparse input views.
Our method enriches each Gaussian with compact semantic embeddings learned via a novel Distilled Prototype Contrastive Loss, combined with 3D intra- and inter-class regularization to enforce local semantic consistency in 3D space. To adaptively allocate representational capacity, we propose a learnable semantics-aware density control strategy that dynamically splits, scales, or removes Gaussians based on semantic region and uncertainty. Furthermore, we mitigate sparse supervision with a consistency-aware pseudo-view generator that uses geometry-conditioned ControlNet diffusion to refine intermediate renders without hallucinating content, providing view-consistent pseudo-supervision. Extensive experiments on three diverse datasets demonstrate that our method significantly outperforms recent few-shot baselines, delivering sharper object boundaries, more accurate semantic segmentation, and higher-fidelity novel view synthesis, particularly under sparse-view constraints.*


