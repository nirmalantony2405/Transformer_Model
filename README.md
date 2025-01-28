.
├── README.md
├── model_evaluation
│  ├──greedy_decode.py
│  └── evaluation.py
├── tests
├── transformer
│  ├── data
│	  ├── dataset.py
│     └── main.py
│  ├── layers
│	  ├── feedforward.py
│	  ├── multi_head_attention.py
│	  ├── transformer_decoder.py
│	  └── transformer_encoder.py
│  ├── modelling
│	  ├── model.py
│	  └── transformer_model.py
│  ├── schedulers
│	  ├── LR_scheduler.py
│     └── adamw_optimizer.py
│  ├── tokenization
│	  ├── bpe_tokenizer.py
│     ├── hf_bpe_tokenizer.py
│	  └── test_tokenizer.py
│  └── training 
│     ├── transformer_model_training.py
│	  └── GPU training.py
└── requirements.txt