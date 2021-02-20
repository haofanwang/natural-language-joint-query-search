# natural-language-joint-query-search
Search photos on Unsplash, including text-to-image search, image-to-image search, text+text-to-image search and search with joint image+text queries.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/haofanwang/natural-language-joint-query-search/blob/main/colab/unsplash-image-search.ipynb)

Search photos on Unsplash using natural language descriptions. The search is powered by OpenAI's [CLIP model](https://github.com/openai/CLIP) and the [Unsplash Dataset](https://unsplash.com/data). This project is mostly based on [natural-language-image-search](https://github.com/haltakov/natural-language-image-search).

### Text-to-Image

![Search results for "Tokyo tower at night."](images/example-text-image-1.png)

![Search results for "People come and go on the street."](images/example-text-image-2.png)

### Image-to-Image

![Search results for a street view image](images/example-image-image-1.png)


### Text+Text-to-Image

![Search results for "flower" and "blue sky"](images/example-text2-image-1.png)

![Search results for "flower" and "bee"](images/example-text2-image-2.png)


### Image+Text-to-Image

![Search results for an empty street](images/example-image+text-image-1.png)


## Acknowledgements

This project was inspired by these projects:

- [OpenAI's CLIP](https://github.com/openai/CLIP)
- [natural-language-image-search](https://github.com/haltakov/natural-language-image-search)
