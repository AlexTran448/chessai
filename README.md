# Fork of ChessAI - Chinese Chess Game Analyzer

# Added features
Added button to start aligning the board
Added Deep learning model for chess board detection (No need to use ARUCO markers)


# Method

- Find estimate of board using Yolov8 model
- Use estimation to find an initial alignment of the board
- Binarise using canny image
- Hide pieces from the image to avoid
- Use Hough transformation to find the lines
- Group lines that are close together (likely to be part of the same line)
- Finding intersections of each line
- Loop through arbitarily chosen intersections (created by 2 vertical and 2 horizontal lines) and compare to other intersections
- Select the best 4 intersections that leads to the largest number other intersections matching the estimated board
- Use intersections to create new more accurate alignment

# Why not use only Yolov8
Few datasets are avaliable, making it difficult to make the model perfect.
Data augmentation was used to give the model more robustness against rotation and other transformations.
Despite this, there were few lighting situations where using Yolo produced poor results




# TODO

[ ] Generate more diverse dataset to allow for more board types to be used
[ ] Automatically check when board has been moved and automatically start aligning the board
[ ] Change model for detecting pieces to use the whole piece instead of just the piece's character (When board is placed at a low angle, the piece face does not align with the correct position, this solution will fix it)

# Weaknesses
**Problem**: Board detection model performs poorly against board that are far from camera
- **Likely cause**: dataset contained boards that were all close to the board
- **Solution**: Include datasets where the board is at far distances from the camera or augment data to change the board's size

**Problem**: Board detection model falsely detects objects such as chessboards
- **Likely cause**: dataset didn't contain many null examples
- **Solution**: Incluide chessboards, and other objects that were commonly missclassified to training set
# -------------------

## Upstream README.md at start of fork


ChessAI is a groundbreaking tool that brings together computer vision, chess algorithms, and advanced analytics to revolutionize the Chinese Chess analytics landscape. With ChessAI, you don't need expensive electronic boards to analyze your games. Simply use your regular board, set up a camera to capture the position, and let ChessAI do the rest.

- Main source code: `chesssai`.
- Deep Learning / Data Preparation: `dnn_models/data_preparation` - Currenly only support for Chinese Chess (XiangQi), [contact me](https://aicurious.io/contact) for the license and the source code of the data preparation tool.
- Deep Learning / Training: `dnn_models/training`.

![ChessAI](https://github.com/vietanhdev/chessai/raw/main/docs/images/screenshot.png)

## Roadmap

- [x] Chess position detection.
- [x] Chess engine integration.
- [x] Move suggestion.
- [ ] Deep learning model for chess board detection (No need to use ARUCO markers).

## Environment setup

- Requirements: Python 3.9, [Conda](https://docs.conda.io/en/latest/miniconda.html), Node.js 18+.
- Clone this repository.

```bash
git clone https://github.com/vietanhdev/chessai --recursive
```

- Create a new conda environment and the required packages.

```bash
conda create -n chessai python=3.9
conda activate chessai
pip install -e .
```

- Install Node.js packages and build the frontend.

```bash
cd chessai/frontend
npm install
cd ..
bash build_frontend.sh
```

## Build chess engine

- This project uses [godogpaw](https://github.com/hmgle/godogpaw) as the chess engine.
- Install [Go](https://go.dev/doc/install).
- Build the engine.

```bash
cd godogpaw
go build
```

- Copy the executable file (`godogpaw*`) to the [./data/engines](./data/engines) folder.

## Run the app

```bash
ENGINE_PATH="data/engines/godogpaw-macos-arm" python -m chessai.app --run_app
```

Replace `ENGINE_PATH` with the path to the chess engine executable file.

## Data preparation & Training

This project uses computer vision and deep learning to detect chess pieces and chess board position.

**AI flow for chess position detection:**

![AI flow for chess position detection](https://raw.githubusercontent.com/vietanhdev/chessai/main/docs/images/ai_flow.png)

- Go to [dnn_models](./dnn_models) folder and follow the instructions in the `README.md` file to prepare the data and train the model.
- **NOTE:** Only training source code and pretrained models are included in this repository. The data preparation scripts and the training datset are not included. [Contact me](https://aicurious.io/contact) for the license and the data.

## References

- This project was initially built for [Hackster's OpenCV AI Competition 2023](https://www.hackster.io/contests/opencv-ai-competition-2023). Hackster Project: [ChessAI - Chinese Chess Game Analyzer](https://www.hackster.io/vietanhdev/chessai-chinese-chess-game-analyzer-4be768).
- Object detection model (for chess pieces) is based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) - License: Apache 2.0.
- Chess engine: [godogpaw](https://github.com/hmgle/godogpaw) - License: MIT.
- UI components: [shadcn-ui](https://github.com/shadcn-ui/ui) - License: MIT.
