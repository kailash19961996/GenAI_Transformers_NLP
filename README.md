# Tiny Story Generator

This project is a transformer-based model that generates short, creative stories from the TinyStories dataset. The goal of this project is to explore the capabilities of transformer models in generating coherent and engaging narratives.

## Features

- Transformer-based architecture for text generation
- Trained on the TinyStories dataset
- Ability to generate stories of varying lengths, starting from a given word and ending with a specified word
- Deployed as a web application using Streamlit, allowing users to interact with the model and generate stories

## Usage

1. **Generate a Story**:
   - Specify a starting word and an ending word (e.g., "she" and "</sos>")
   - Provide a maximum length for the generated story
   - The model will generate a story, starting from the provided word and ending with the specified word (or reaching the maximum length)

2. **Interactive Web Application**:
   - The project is deployed as a web application using Streamlit, allowing users to interact with the model and generate stories directly in their web browsers.
   - Users can input the starting and ending words, as well as the maximum length, and the generated story will be displayed on the page.

## Technical Details

The model is built using PyTorch and the Transformer architecture. It consists of:

- A Decoder layer with Multi-Head Attention and Position-Wise Feed-Forward components
- Positional Encoding to incorporate sequence information
- An Embedding layer to map tokens to dense vector representations
- A Linear layer to project the output to the target vocabulary size

The model is trained on the TinyStories dataset, with the objective of minimizing the cross-entropy loss between the model's output and the target sequence.

## Architecture

```python
Transformer(
  (decoder_embedding): Embedding(5000, 512)
  (positional_encoding): PositionalEncoding()
  (decoder_layers): ModuleList(
    (0-5): 6 x DecoderLayer(
      (self_attn): MultiHeadAttention(
        (W_q): Linear(in_features=512, out_features=512, bias=True)
        (W_k): Linear(in_features=512, out_features=512, bias=True)
        (W_v): Linear(in_features=512, out_features=512, bias=True)
        (W_o): Linear(in_features=512, out_features=512, bias=True)
      )
      (feed_forward): PositionWiseFeedForward(
        (fc1): Linear(in_features=512, out_features=512, bias=True)
        (fc2): Linear(in_features=512, out_features=512, bias=True)
        (relu): ReLU()
      )
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (fc): Linear(in_features=512, out_features=5000, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)'''


## Future Improvements

- Explore different techniques to improve the coherence and creativity of the generated stories, such as incorporating additional context or using more advanced generation methods.
- Experiment with different model architectures or training strategies to enhance the model's performance.
- Expand the application functionality, such as allowing users to save or share the generated stories, or providing more customization options.

## Deployment

The Tiny Story Generator is deployed as a web application using Streamlit

## Acknowledgments

This project was inspired by the advancements in transformer-based language models and the TinyStories dataset. Big thanks to the contributions of the research community in the field of natural language processing and generation.
