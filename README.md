# Automated_VT_Localisation

If using any of the scripts in this repository, please cite

Sofia Monaci, Shuang Qian, Karli Gillette, Esther Puyol-Antón, Rahul Mukherjee, Mark K Elliott, John Whitaker, Ronak Rajani, Mark O’Neill, Christopher A Rinaldi, Gernot Plank, Andrew P King, Martin J Bishop, Non-invasive localization of post-infarct ventricular tachycardia exit sites to guide ablation planning: a computational deep learning platform utilizing the 12-lead electrocardiogram and intracardiac electrograms from implanted devices, EP Europace, Volume 25, Issue 2, February 2023, Pages 469–477, https://doi.org/10.1093/europace/euac178

## ABOUT

This platform allows to test a pre-trained DL architecture on a scar-related VT ECG/EGM trace to predict corresponding exit site. There is also the possibility to re-train the first part of the pipeline after adding new pacing simulations to the existing pacing dataset. The network then needs to be re-trained on the simulated scar-related VT traces

This repository also contains files to generate virtual, simplistic scars around a mesh of interest.
