# Video Processing Project

This project is a web application for uploading and processing videos with various effects using SAM2 and OpenCV.

## Features

- Upload videos
- Track a given object throughout the video
- Apply different effects to the video background and objects
- Display processed videos

## Requirements

- Python 3.x
- Django 3.2.9
- NumPy 1.21.2
- OpenCV 4.5.3.56
- SAM2 0.4.1

## Installation

1. Clone the repository:
    ```sh
    git clone git@github.com:ruiczhu/videoProcessWithSam2.git
    cd videoProcessWithSam2
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   
## Contributing

1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature-branch
    ```
3. Commit your changes:
    ```sh
    git commit -am 'Add new feature'
    ```
4. Push to the branch:
    ```sh
    git push origin feature-branch
    ```
5. Create a new Pull Request.

## Citing SAM 2
- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2\
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}