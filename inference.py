import torch
import torchaudio
from src.model import KWSNet
from config import params
from src.dataset import transforms
import wandb

device = torch.device('cpu')
model = KWSNet(params)
model.load_state_dict(torch.load(params["model_path"], map_location=device))
model = model.eval()
wav_file = params["path_to_file"]
wav, sr = torchaudio.load(wav_file)
input = transforms['test'](wav)
probs = model.inference(input, params["window_size"])
wandb.init(project=params["wandb_name"], config=params)
for prob in probs:
    wandb.log({"probs two trigger words in the rain": prob})
wandb.log({"test audio": [wandb.Audio(wav.numpy(), caption="Sheila sheila", sample_rate=22050)]})

