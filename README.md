# Self-Supervised Blind Source Separation via Multi-Encoder Autoencoders

## Methodology Overview
We propose a novel methodology for addressing blind source separation of non-linear mixtures via multi-encoder single-decoder autoencoders with fully self-supervised learning. During training, our methodology unmixes the input into the multiple encoder output spaces and then remixes these representations within the single decoder for a simple reconstruction of the input. Then to perform source inference we introduce a novel encoding masking technique whereby masking out all but one of the encodings enables the decoder to estimate a source signal. To achieve consistent source separation, we also introduce a so-called _pathway separation loss_ for the decoder that encourages sparsity between the unmixed encoding spaces throughout and a so-called _zero reconstruction loss_ on the decoder that assists with coherent source estimations. We conduct experiments on a toy dataset, the _triangles & circles_ dataset, and with real-world biosignal recordings from a polysomnography sleep study for extracting respiration.

<img src="assets/bss_graph_1.png" alt="drawing" width="30%" height="30%"/>

<img src="assets/bss_graph_2.png" alt="drawing" width="30%" height="30%"/>


### Getting Started with the _triangles & circles_ dataset
1. To generate your own dataset please see: [notebooks/triangles_and_circles_dataset.ipynb](notebooks/triangles_and_circles_dataset.ipynb)
2. To train a model with our configuration use the following command: `python trainer.py experiment_config=tri_and_circ_bss`
3. Lastly, to test your model please see: [notebooks/triangles_and_circles_test_model.ipynb](notebooks/triangles_and_circles_test_model.ipynb)



### Citation
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