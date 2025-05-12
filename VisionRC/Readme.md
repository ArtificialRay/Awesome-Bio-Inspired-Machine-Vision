VisionRC is an image recognition network combines the concept of reservoir computing(RC), graph neural network(GNN), and general skillset in deep neural network for machine vision. 

VisionRC has isotropic architecture (VisionRC.py) inspired by ViT and pyramid architecture (pVisionRC.py) inspired by CNN. A typical VisionRC block is formed by:

<img width="831" alt="image" src="https://github.com/user-attachments/assets/5bd386d5-ff72-410f-8701-3e93563a379a" />

Graph processing unit is conducted by a new mechanism "ResGraph": Using RC to update node feature in a given graph:

<img width="677" alt="image" src="https://github.com/user-attachments/assets/443adb26-7de4-4ccd-bf70-f67d26bc8b90" />

The project apply CIFAR-10 dataset to evaluate model performance, compare to Vision GNN (ViG) model proposed by Han et.al [[arXiv link]](https://arxiv.org/abs/2206.00272)

<img width="310" alt="image" src="https://github.com/user-attachments/assets/92a5cade-d4bc-43c8-80c1-0d7519b723e3" />

VisionRC has a competitive recognition performance compared with ViG with less parameter size and FLOPs. Note that Params and FLOPs are measured under 32x32 resolution:
- Isotropic architecture
  
|Model|Depth|Params (M)|FLOPs (B)|Test accuracy|
|-|-|-|-|-|
|VisionRC|6|2.94|63.63|71%|
|ViG - Ti|12|6.13|97.07|72%|

- Pyramid architecture
  
|Model|Params (M)|FLOPs (B)|Test accuracy|
|-|-|-|-|
|Pyramid VisionRC|3.21|0.36|84%|
|Pyramid ViG - Ti|9.79|0.52|86%|

