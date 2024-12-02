# Categorized Triton Functions

This repository contains a bunch of Triton functions that have been categorized into different categories using Anthropic's Claude 3.5 Sonnet model. You can find them under the `sorted_functions` folder.

To run the script yourself, you can do the following:

```bash
cp .env.example .env
# fill out .env with your api Anthropic API key
python categorize_triton.py
# take a look at the generated plot and `sorted_functions` folder
```

Todo: 
 - [ ] Currently all the funcitons are stored in `github_triton.json`. We should 1 create a script to regenerate it. It's simply composed of searching for triton functions in any repo which 1) mentions "triton" in the readme, 2) contains `@triton.jit` in the code, and 3) has 5+ stars.
 - [ ] Use an arbitrary datasource instead of `github_triton.json`.

Acknowledgements:
- This project was inspired by [bookvis](https://github.com/n-e-w/bookvis).