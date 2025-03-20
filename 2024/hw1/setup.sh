# Modify this command depending on your system's environment.
# As written, this command assumes you have CUDA on your machine, but
# refer to https://pytorch.org/get-started/previous-versions/ for the correct
# command for your system.
# For OSX, use `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0`
# For Linux and Windows (GPU version), use `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`
# For Linux and Windows (CPU version), use `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu`
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==1.2.2
pip install numpy==1.26.3
pip install tokenizers==0.13.3
pip install sentencepiece==0.1.99
