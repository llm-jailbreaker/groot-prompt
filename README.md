# Groot
 The first automated framework leveraging tree-based semantic transformation for adversarial testing of text-to-image models.

 Text-to-image generative models, such as DALL-E 3, Midjourney, and Stable Diffusion, which transform textual descriptions into visual representations, are increasingly prevalent in a variety of applications, making their safety a critical concern. To assess the safety of these models, adversarial testing techniques have been developed to probe whether such models can be prompted to produce Not-Safe-For-Work (NSFW) content. Despite these efforts, current methodologies face several challenges: (1) they exhibit a relatively low success rate in eliciting NSFW content, (2) many rely on random methods of perturbing prompts without a deep semantic understanding, rendering them inefficient, and (3) they tend to focus solely on circumventing safety filters, which often results in output that deviates significantly from the original adversarial intent, even when they successfully evade all filters.

In this work, we introduce Groot, to the best of our knowledge, the first automated framework leveraging tree-based semantic transformation for adversarial testing of text-to-image models. Groot employs semantic decomposition and sensitive element drowning strategies in conjunction with Large Language Models (LLMs) to systematically refine adversarial prompts. Our comprehensive evaluation confirms the efficacy of Groot, which not only exceeds the performance of current state-of-the-art approaches but also achieves a remarkable success rate 93.66% on leading text-to-image models such as DALL-E 3 and Midjourney.

## Dataset

We present our evaluation data NSFW-363 [here](https://drive.google.com/file/d/1pC1y_YUBaSaCzJNeweSt2ffhBA1hOL-l/view?usp=sharing),  please send the author an email for password.

Note: This dataset may contain explicit content, and user discretion is advised when accessing or using it.

- Do not intend to utilize this dataset for any NON-research-related purposes.

- Do not intend to distribute or publish any segments of the data.



## Experiments

Before runnng, you should replace the string 'YOUR_OPENAI_API_KEY' in the source code with an openai api key that works.

We use the small dataset `data/example.txt` as example.

The results is stored as `.csv` file with fields:
- `sample` : the content of the sample text.
- `attack` : the number of attacks when succeed.
- `one-layer-leaves` : the number of leaves in one layer.
- `fail` : whether the attack fails.
- `success` : whether the attack succeeds.
- `content-filters` : whether the input text passes text filter but generated image is filtered by the content filter.

To evaluate Groot, run the `scripts/Groot-1.0.py`.

```shell
python3 scripts/Groot-1.0.py data/example.txt logs/groot.log restults/groot.csv
``` 

To evaluate semantic decompsition, run the `scripts/SemanticDecomposition-1.0.py`.

```shell
python3 scripts/SemanticDecomposition-1.0.py data/example.txt logs/semantic_decomposition.log restults/semantic_decomposition.csv
``` 

To evaluate drown, run the `scripts/Drown-1.0.py`.

```shell
python3 scripts/Drown-1.0.py data/example.txt logs/drown.log restults/drown.csv
``` 