# Abstract

As the volume of text-based data continues to rapidly increase, the requirement for efficient compression methods is greater than ever before. The advancements in Natural Language Processing (NLP) provide a golden opportunity to employ semantically-driven methods for text compression. Our project aspires to extend the boundaries of text compression by integrating a transformer-based neural network with two distinct data compression techniques: variable-length integer encoding and arithmetic encoding. Preliminary results from our project indicate that our neural text compression strategy achieves a compression ratio twice that of the industry-standard Gzip. Despite the computational time for our neural-based compression being notably longer compared to other baselines, we've discovered that this can be minimized by implementing a windowing approach, thereby reducing the impact of high self-attention computations.

# Approach

Our method involves feeding our transformer-based model with a document, word-by-word, to compress. For instance, when processing word wi, the model is given the previous k′ words, w_i−k′, ..., w_i−2, w_i−1, words in a sequence and is tasked with generating a vocabulary distribution for the prediction of the next word, where k′ denotes the number of words preceding wi in the current window. This next-word distribution is organized from the highest probability to the lowest. We continuously iterate over these next-word options until we identify w_i. The position of this word in the sorted list is subsequently processed through either our variable-length integer encoding algorithm or our arithmetic encoding algorithm, and this encoded information forms a part of the compressed data. We chose the pretrained GPT2 model as our preferred transformer due to its highly effective "next-word" prediction capability that aligns seamlessly with our needs, and we further fine-tune it on the task of text generation using the wikitext2 dataset.

<div align="center">
<img  src="src/img/framework.png"  align = 'center' width="700">
  <figcaption><br>Compression Framework</figcaption>
</div>
