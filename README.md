# llm-paper-data-retrieval

A repository to experiment with retrieving key data from ML research papers, using OpenAI GPT models.

Currently working with a sample of 18 papers to test this.

File reference:
- `save_paper_txts.py`: run this first to download the sample of papers as txt files.
  - There is commented-out code in the notebook to download the papers on the fly, but currently the TXT files are used to save time.
- `paper_data_retrieval.ipynb`: the Python notebook where the main experiment is implemented.
- `notes.md`: detailed notes on the experiment process. This can help with reproducibility but is not recommended to read because it is very long and messy.
