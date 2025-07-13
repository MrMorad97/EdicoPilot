import matplotlib.pyplot as plt
import numpy as np

# Turns
turns = np.arange(1, 9)

# BERTScores
gpt2_scores = [0.7911, 0.8027, 0.7778, 0.8042, 0.8429, 0.8221, 0.8331, 0.8515]
ft_gpt2_scores = [0.7582, 0.7443, 0.7593, 0.8318, 0.8866, 0.8218, 0.8136, 0.8216]
custom_lm_scores = [0.8858, 0.8088, 0.8239, 0.8282, 0.8494, 0.8143, 0.8454, 0.8253]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(turns, gpt2_scores, marker='o', label='GPT-2 (pretrained)')
plt.plot(turns, ft_gpt2_scores, marker='s', label='Fine-tuned GPT-2')
plt.plot(turns, custom_lm_scores, marker='^', label='Custom LM')

# Formatting
plt.title('BERTScore F1 per Turn for Dialogue Models')
plt.xlabel('Turn')
plt.ylabel('BERTScore F1')
plt.xticks(turns)
plt.ylim(0.7, 0.9)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
