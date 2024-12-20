# Sample Search 🎧

An AI-powered semantic search engine for music samples that allows you to find audio samples using natural language descriptions. The system uses the CLAP model (Contrastive Language-Audio Pretraining) to understand both audio content and text descriptions, enabling semantic search capabilities.

**NOTE**: The current implementation is compatible *only* with MacOS. Minor adjustments are required for Windows and Linux compatibility, which will be addressed soon.

## Demo
https://www.loom.com/share/f41e5a06b34e448899467bd3a7bae744?sid=cee4ebdb-a03e-496e-8bfe-35c8f3f91010


## Features

- Index all your audio samples to a vector database.
- Semantic search using natural language descriptions.

## Requirements

- Python 3.11+
- Poetry 1.8.3+
- Docker
- Docker Compose
- libsndfile

## How to use

1. Clone the repository.
```bash
git clone https://github.com/dariowsz/sample-search
cd sample-search
```

2. Create virtual environment and install dependencies.
```bash
python -m venv .venv
source .venv/bin/activate
poetry install --only=main
```

3. Install PyTorch manually.
PyTorch installation is separated from the rest of the dependencies because it requires a different set of commands for each operating system and hardware architecture. If you are using MacOS, you can install it using the following command:
```bash
pip install -r requirements.macos.txt
```

4. Start the vector database with Docker Compose.
```bash
docker compose up -d
```

5. [OPTIONAL] Change the environment variables. I pushed them because there are no secrets here.


6. Run the application.
```bash
streamlit run src/demo.py
```

## Future Work

- [ ] Add a way to search samples by audio.
- [ ] Compatibility with other OS (Windows and Linux)
- [ ] Improve the UI/UX.
- [ ] Add new encoder models and improve existing ones.
- [ ] Finish Swift application with CoreML models (once Apple allows it).
