# Bible Chapters Embedding Visualization

This project aims to visualize Bible chapters in a 3D space by embedding their textual content using OpenAI's text-embedding-ada-002 model, then applying t-SNE for dimensionality reduction.

## Getting Started

### Installing

1. Clone the repository:
   git clone https://github.com/dominicpeel/bible-chapters-embedding.git

2. Change directory into the project folder:
   cd bible-chapters-embedding

3. Install the required packages:
   pip install -r requirements.txt

4. Set up your OpenAI API key in a `.env` file:
   echo "OPENAI_API_KEY=your_api_key" > .env

## Running the project

Run the create_embeddings to generate the embeddings of Bible chapters:

```bash
python create_embeddings.py
```

Run the embeddings_vis.py to visualize the embeddings in a 3D scatter plot using t-SNE dimensionality reduction:

```bash
python embeddings_vis.py
```

The script will output a 3D scatter plot of the Bible chapters in your browser.

## Output

The project outputs the following files:

- `embeddings.csv`: A CSV file containing the embeddings for each chapter.
- `embeddings_3d.csv`: A CSV file containing the reduced 3D coordinates for each chapter.

## License

MIT, of course :)
