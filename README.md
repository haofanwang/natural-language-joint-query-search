# natural-language-joint-query-search

In the project, we support multiple types query search including text-image, image-image, text2-image and text+image-image. In order to analyse the result of retrieved images, we also support visualization of CLIP attention.

## Colab Demo

Search photos on Unsplash, support for joint image+text queries search.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haofanwang/natural-language-joint-query-search/blob/main/colab/unsplash_image_search.ipynb)

Attention visualization of CLIP.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haofanwang/natural-language-joint-query-search/blob/main/colab/clip_attention.ipynb)

## Example

### Text-to-Image

##### "Tokyo tower at night."
![Search results for "Tokyo tower at night."](images/example-text-image-1.png)

##### "People come and go on the street."
![Search results for "People come and go on the street."](images/example-text-image-2.png)

### Image-to-Image

##### A normal street view. (The left side is the source image)
![Search results for a street view image](images/example-image-image-1.png)


### Text+Text-to-Image

##### "Flower" + "Blue sky"
![Search results for "flower" and "blue sky"](images/example-text2-image-1.png)

##### "Flower" + "Bee"
![Search results for "flower" and "bee"](images/example-text2-image-2.png)


### Image+Text-to-Image

##### A normal street view + "cars"
![Search results for an empty street with query "cars"](images/example-image+text-image-1.png)


## Acknowledgements

Search photos on Unsplash using natural language descriptions. The search is powered by OpenAI's [CLIP model](https://github.com/openai/CLIP) and the [Unsplash Dataset](https://unsplash.com/data). This project is mostly based on [natural-language-image-search](https://github.com/haltakov/natural-language-image-search).

This project was inspired by these projects:

- [OpenAI's CLIP](https://github.com/openai/CLIP)
- [natural-language-image-search](https://github.com/haltakov/natural-language-image-search)
