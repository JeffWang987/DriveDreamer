<div align="center">   

# DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving
</div>

## [Project Page](https://drivedreamer.github.io) | [Paper](https://drivedreamer.github.io/)

# Abstract 

World models, especially in autonomous driving, are trending and drawing extensive attention due to its 
capacity for comprehending driving environments. The established world model holds immense potential 
for the generation of high-quality driving videos, and driving policies for safe maneuvering. However, 
a critical limitation in relevant research lies in its predominant focus on gaming environments or simulated 
settings, thereby lacking the representation of real-world driving scenarios. Therefore, we introduce 
DriveDreamer, a pioneering world model entirely derived from real-world driving scenarios. Regarding that 
modeling the world in intricate driving scenes entails an overwhelming search space, we propose harnessing 
the powerful diffusion model to construct a comprehensive representation of the complex environment. Furthermore, 
we introduce a two-stage training pipeline. In the initial phase, DriveDreamer acquires a deep understanding of 
structured traffic constraints, while the subsequent stage equips it with the ability to anticipate future states. 
The proposed DriveDreamer is the first world model established from real-world driving scenarios. We instantiate 
DriveDreamer on the challenging nuScenes benchmark, and extensive experiments verify that DriveDreamer empowers precise,
controllable video generation that faithfully captures the structural constraints of real-world traffic scenarios.  
Additionally, DriveDreamer enables the generation of realistic and reasonable driving policies, opening avenues for 
interaction and practical applications.
<img width="907" alt="abs" src="https://github.com/JeffWang987/DriveDreamer/assets/49095445/c2d7fb5b-75f1-4a97-9940-cdd49e7675c8">



# News
- **[2023/09/17]** Repository Initialization.


# Demo
**Driving Video Generation with Different Traffic Conditions.**
<div align="center">   

https://github.com/JeffWang987/DriveDreamer/assets/49095445/8b17ab65-2c18-4a83-9f35-38ad29f9bce1
      
</div>


**Driving Video Generation with Traffic Condition and Different Text Prompts (Sunny, Rainy, Night).**

<div align="center">   


https://github.com/JeffWang987/DriveDreamer/assets/49095445/162c31cf-8dd2-4eed-b585-f346d58b714f


</div>


**Future Driving Video Generation with Action Interaction.**

<div align="center">   


https://github.com/JeffWang987/DriveDreamer/assets/49095445/e96e02b1-4a0a-4bc5-9221-d57d1947ceaf


</div>

**Future Driving Action Generation.**

<div align="center">   


https://github.com/JeffWang987/DriveDreamer/assets/49095445/b6893c6c-5137-4270-8fe3-b4d1668b80e8


</div>


**DriveDreamer Framework**

<img width="1314" alt="method" src="https://github.com/JeffWang987/DriveDreamer/assets/49095445/ad395b96-0696-4118-b794-3e34469955bd">


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2023drivedreamer,
      title={DriveDreamer: Towards Real-world-driven World Models for Autonomous  Driving}, 
      author={Xiaofeng Wang and Zheng Zhu and Guan Huang and Xinze Chen},
      journal={arXiv preprint arXiv:TODO},
      year={2023}
}
```

