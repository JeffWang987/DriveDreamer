<div align="center">   

# DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving
</div>
 
## [Project Page](https://drivedreamer.github.io) | [Paper](https://arxiv.org/pdf/2309.09777.pdf)

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
<img width="907" alt="abs" src="https://github.com/JeffWang987/DriveDreamer/assets/49095445/9e3829cf-c24b-4f96-a75e-37508b4aead7">




# News
- **[2024/11/02]** 🚀 Research code for DriveDreamer is realsed! For future update, please follow [GigaAI-research](https://github.com/GigaAI-research).
- **[2024/10/17]** 🚀We release the [DriveDreamer4D](https://drivedreamer4d.github.io/) project! (Key features: 4D reconstruction, complex maneuvers rendering)
- **[2024/07/02]** 🎉DriveDreamer is accepted for ECCV'24! Rank 15 in  [Most Influential ECCV'24 Papers](https://www.paperdigest.org/2024/09/most-influential-eccv-papers-2024-09/).
- **[2024/03/11]** 🚀We release the [DriveDreamer-2](https://drivedreamer2.github.io/) project! (Key features: multi-view video generation, user-friendly with LLM)
- **[2023/09/17]** 🚀We release the [DriveDreamer](https://drivedreamer.github.io/) project!



# Getting Started

- [Installation](DOCS/install.md) 

- [Prepare Dataset & Env](DOCS/preparation.md)

- [Train, Test, Visualization](DOCS/trainval.md)


# Demo
**Diverse Driving Video Generation.**
<div align="center">   



https://github.com/JeffWang987/DriveDreamer/assets/49095445/a1f658ff-3ddc-4ec8-9e1f-9d3fe7183350


      
</div>


**Driving Video Generation with Traffic Condition and Different Text Prompts (Sunny, Rainy, Night).**

<div align="center">   




https://github.com/JeffWang987/DriveDreamer/assets/49095445/9cdf8e59-08bd-4c09-980c-2a66b0c0c0b8




</div>


**Future Driving Video Generation with Action Interaction.**

<div align="center">   



https://github.com/JeffWang987/DriveDreamer/assets/49095445/14133f36-f557-47f5-b7cd-ecdb0c76f050




</div>

**Future Driving Action Generation.**

<div align="center">   


https://github.com/JeffWang987/DriveDreamer/assets/49095445/b6893c6c-5137-4270-8fe3-b4d1668b80e8


</div>


**DriveDreamer Framework**


<img width="1340" alt="method" src="https://github.com/JeffWang987/DriveDreamer/assets/49095445/9d578df7-780d-4518-a1e5-f2e030d7df7e">


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2023drivedreamer,
  title={Drivedreamer: Towards real-world-driven world models for autonomous driving},
  author={Wang, Xiaofeng and Zhu, Zheng and Huang, Guan and Chen, Xinze and Zhu, Jiagang and Lu, Jiwen},
  journal={arXiv preprint arXiv:2309.09777},
  year={2023}
}
```

