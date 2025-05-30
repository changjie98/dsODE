This project provides simulation code associated with the publication
[Minimizing information loss reduces spiking neuronal networks to differential equations](https://arxiv.org/abs/2411.14801).

# Model Description
The dsODE model is written in Matlab and does not require any additional packages. The refactory density methods and the Fokker-Planck equations used for comparison with the dsODE model in the project are written in Python, and specific operating conditions can be referenced from the original author's project.

# Examples
> main_LIF  
> main_dsODE  
> plot(0.1:0.1:1000,smooth(res_lif.fr_e,10),'r','linewidth',1)  
> plot(0.1:0.1:20000,smooth(res_dsODE.fr_e,10),'b','linewidth',1)  
