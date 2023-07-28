# Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders

## Methodology Overview
We propose a novel methodology for addressing blind source separation of non-linear mixtures via multi-encoder single-decoder autoencoders with fully self-supervised learning. During training, our methodology unmixes the input into the multiple encoder output spaces and then remixes these representations within the single decoder for a simple reconstruction of the input. Then to perform source inference we introduce a novel encoding masking technique whereby masking out all but one of the encodings enables the decoder to estimate a source signal. To achieve consistent source separation, we also introduce a so-called _pathway separation loss_ for the decoder that encourages sparsity between the unmixed encoding spaces throughout and a so-called _zero reconstruction loss_ on the decoder that assists with coherent source estimations. We conduct experiments on a toy dataset, the _triangles & circles_ dataset, and with real-world biosignal recordings from a polysomnography sleep study for extracting respiration.
<p align="center">
    <img src="assets/bss_graph_1.png" alt="drawing" width="50%" height="50%"/>
</p>
<p align="center">
    <img src="assets/bss_graph_2.png" alt="drawing" width="50%" height="50%"/>
</p>

## Experiments
### Triangles & Circles
#### 1. Getting Started with the _triangles & circles_ dataset
1. To generate your own dataset please see: [notebooks/triangles_and_circles_dataset.ipynb](notebooks/triangles_and_circles_dataset.ipynb)
2. To train a model with our configuration use the following command: `python trainer.py experiment_config=tri_and_circ_bss`
3. Lastly, to test your model please see: [notebooks/triangles_and_circles_test_model.ipynb](notebooks/triangles_and_circles_test_model.ipynb)

#### 2. Training Demo

<p align="center">
    <img src="assets/training_demo.gif" alt="drawing" width="35%" height="35%"/>
</p>

### ECG & PPG Respiratory Source Extraction


#### Results
<p align="center">
    <img src="assets/ppg.png" alt="drawing" width="60%" height="60%"/>
</p>
<p align="center">
    <img src="assets/ecg.png" alt="drawing" width="60%" height="60%"/>
</p>

We evaluate our methodology by extracting respiratory rate from the estimated source (manually reviewed to correspond with respiration) and comparing it the extracted respiratory rate of a simultaneously measured reference respiratory signal, nasal pressure or thoracic excursion.

| Method (Type)       | Breaths/Min. MAE $\downarrow$ | Breaths/Min. MAE $\downarrow$ | Method (Type)                 | Breaths/Min. MAE $\downarrow$ | Breaths/Min. MAE $\downarrow$ |
|---------------------|------------------------------|------------------------------|---------------------------------|-------------------------------|-------------------------------|
| **BSS**             | **Nasal Press.**             | **Thor.**                    | **Heuristic**                   | **Nasal Press.**              | **Thor.**                     |
|                     |                              |                              |                                 |                               |                               |
| Ours (PPG)          | 1.51                         | 1.50                         | (Muniyandi & Soni, 2017)  (ECG) | 2.38                          | 2.04                          |
| Ours (ECG)          | 1.73                         | 1.59                         | (Charlton et al., 2016) (ECG)   | 2.38                          | 2.05                          |
|                     |                              |                              | (van Gent et al., 2019) (ECG)   | 2.27                          | 1.95                          |
|                     |                              |                              | (Sarkar, 2015)  (ECG)           | 2.26                          | 1.94                          |
| **Supervised**      |                              |                              |  **Direct Comparison**          |                               |                               |
|                     |                              |                              |                                 |                               |                               |
| AE (PPG)            | 0.46                         | 2.07                         | Thor.                           | 1.33                          | --                            |
| AE (ECG)            | 0.48                         | 2.16                         |                                 |                               |                               |







### Cite our work
Our work, 'Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders', is currently under review. If you find this repository helpful, please cite us.
```
@software{webster2023,
  author = {Webster, M.B. and Lee Joonnyong},
  title = {Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/webstah/self-supervised-bss-via-mult-encoder-ae}},
}
```