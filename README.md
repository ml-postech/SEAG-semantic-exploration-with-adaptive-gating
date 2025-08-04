# SEAG-semantic-exploration-with-adaptive-gating
[ACL 2025] Official implementation of [Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](https://arxiv.org/pdf/2501.05752)

You can visit [our project page](https://sjaelee25.github.io/seag/) 🚀!

## 🚀 Running the GSM8K Experiment
To run the GSM8K experiment using our implementation:

```bash
TIKTOKEN_CACHE_DIR="" CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1111 \
expample_gsm8k/inference.py \
--base_lm llama-3 --llama_3_ckpts <path_to_llama3_ckpt> --llama_size "8B-Instruct" \
--n_iters 10 --early_term_threshold 11 \
--log_dir ./logs/gsm8k/
```

- Arguments
llama_3_ckpts: (Required) Path to the LLaMA 3 checkpoint


## Acknowledgements

Our implementation builds upon the following repositories:

- [**Reasoning with Language Model is Planning with World Model**](https://arxiv.org/pdf/2305.14992)  
  → Code: [llm-reasoners](https://github.com/maitrix-org/llm-reasoners)

- [**Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation**](https://arxiv.org/pdf/2302.09664)  
  → Code: [semantic-uncertainty](https://github.com/lorenzkuhn/semantic_uncertainty)  
  
## 📬 Contact
If you have any questions, feel free to contact:

- 📧 Hyejin Park: parkebbi2@postech.ac.kr
- 📧 Sungjae Lee: sungjaelee25@postech.ac.kr

# Citation
```
@article{lee2025semantic,
  title={Semantic exploration with adaptive gating for efficient problem solving with language models},
  author={Lee, Sungjae and Park, Hyejin and Kim, Jaechang and Ok, Jungseul},
  journal={arXiv preprint arXiv:2501.05752},
  year={2025}
}
```
