
---

# Numerai Competition Model

![Numerai Logo](https://upload.wikimedia.org/wikipedia/commons/c/ce/Numerai_logo.png)

Welcome to the open-source repository for our Numerai competition model!

## About

This repository contains the code for our model developed for the Numerai competition. The model is designed to predict the target variable with the goal of achieving high performance on the Numerai tournament data.

## Features

- Open-source code
- MIT License

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

To use this model, you need to clone the repository and obtain a secret key for the Numerai API. Please follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Fintanch/Numerai-Competition-Model.git
   cd numerai-competition-model
   chmod +x ./install_dependencies.sh
   conda create -n numerai python=3.8
   conda activate numerai
   ./install_dependencies.sh
   ```

2. Obtain your Numerai secret key and save it to a file named `Numerai.secrets`. The file should be a copy of Numerai.secrets.template and look like this:

   ```
   PUBLIC_KEY=your_public_key_here
   SECRET_KEY=your_secret_key_here
   ```

   Replace `your_public_key_here` and `your_secret_key_here` with your actual Numerai API keys.

## Usage

1. Follow the installation steps to clone the repository and obtain your Numerai secret key.

2. Run the training script:

   ```bash
   python train_model.py
   ```

   Alternatively, you can utilize our Training.ipynb where usage should be documented, and you will have to do your best to follow the steps required.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to reach out to the Authors at:

- Email: frattitamayo@gmail.com
- Email: johnnykoch02@gmail.com

---
