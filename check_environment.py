import sys

def check_imports():
    libraries = [
        'torch', 'torchvision', 'torchaudio', 
        'numpy', 'pandas', 'matplotlib',
        'transformers', 'sklearn', 'tqdm'
    ]
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"{lib} successfully imported")
        except ImportError:
            print(f"Error: {lib} is not installed")

def check_cuda():
    import torch
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. The code will run on CPU.")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    check_imports()
    check_cuda()