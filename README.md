# Presentation

This project is an attempt to extend the work presented in the paper "Compositional attention networks for machine reasoning" [1] to textual input and reasoning

# Installation

To install the required packages, run the following command:

    pip install -r requirements.txt

This project is built upon ParlAI [2]. To train a model on a single GPU, simply run the script `mono_train.sh`. To train on a single host with multiple GPUs, run `multi_train.sh`.  Make sure to modify the variables `output_dir` and `datapath` in the scripts mentioned earlier.

[1] Hudson, D. A. and Manning, C. D. (2018). Compositional attention networks for machine reasoning.
In Proceedings of the International Conference on Learning Representations (ICLR).

[2] Alexander H. Miller, Will Feng, Dhruv Batra, Antoine Bordes, Adam Fisch, Jiasen
Lu, Devi Parikh, and Jason Weston. 2017.
Parlai: A dialog research software platform. In
Proceedings of the Conference on Empirical Methods in Natural Language Processing, EMNLP,
pages 79–84.