params = {"num_features": 40,
          "cnn_channels": 16,
          "cnn_kernel_size": 51,
          "gru_hidden_size": 64,
          "attention_hidden_size": 64,
          "window_size": 30,
          "batch_size": 64,
          "num_workers": 8,
          "lr": 0.001,
          "sample_rate": 16000,
          "num_epochs": 10,
          "noise_variance": 0.01,
          "min_time_stretch": 0.9,
          "max_time_stretch": 1.1,
          "min_shift": -3,
          "max_shift": 3,
          "time_masking": 1,
          "wandb_name": "KWSNet",
          "clip_grad_norm": 15,
          "vocab_size": 120,
          "from_pretrained": False,
          "model_path": "kws_model.pth",
          "start_epoch": 40,
          "path_to_file": "test.wav",
          }