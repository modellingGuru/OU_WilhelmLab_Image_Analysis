# Wilhelm_Lab_Analysis
Image Analysis - NanoParticle Segmentation 
OU Wilhelm Lab — Nanoparticle Image Analysis Pipeline

A Python-based image analysis pipeline developed during undergraduate research at the Wilhelm Lab, University of Oklahoma. The pipeline automates segmentation, quantification, and classification of nanoparticles from fluorescence and brightfield microscopy images, with a focus on Gold Nanoparticles (AuNPs) in tumor spheroid models.


What This Does

StageDescriptionSegmentationIsolates individual nanoparticles from raw microscopy images using custom kernels and morphological operationsQuantificationMeasures nanoparticle distribution, density, and spatial patterns across tissue/spheroid cross-sectionsClassificationCategorizes particles by size, intensity, and localization to support downstream nanomedicine analysis

The work supported research into optimizing nanoparticle delivery for cancer therapy — specifically understanding how AuNPs distribute within 3D tumor spheroid models.


Repository Structure

├── AuNP/          # Custom kernels and segmentation code for Gold Nanoparticle analysis
├── src/           # Core pipeline modules (segmentation, quantification, classification)
├── tests/         # Unit tests


Dependencies

bashpip install numpy opencv-python scikit-image tensorflow matplotlib scipy


ImageJ/FIJI was used for manual annotation and ground-truth labeling. Deep learning components use TensorFlow.




Background

This repository represents personal, unofficial code written alongside research conducted in the Wilhelm Lab. The core methods informed published work on nanoparticle-based cancer diagnostics and treatment. The official lab codebase is proprietary; this repo captures the analytical methods I developed independently.


Author

Michael Marah

B.S. Biomedical Engineering, University of Oklahoma

michaelmarah02@ou.edu
