import torch
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from pathlib import Path

torch.serialization.add_safe_globals([np.ndarray])
history = torch.load("models/training_history.pt", weights_only=False)
epochs = history['epochs']


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
plt.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
plt.title('Loss Curves')
plt.legend(); plt.grid()

plt.subplot(1, 3, 2)
plt.plot(epochs, history['train_accs'], 'b-', label='Train Acc')
plt.plot(epochs, history['val_accs'], 'r-', label='Val Acc')
plt.title('Accuracy Curves')
plt.ylabel('Accuracy %'); plt.legend(); plt.grid()

plt.subplot(1, 3, 3)
plt.bar(['Final Train', 'Final Val'], [history['final_train_acc'], history['final_val_acc']])
plt.title('Final Performance')
plt.ylabel('Accuracy %')

plt.tight_layout()
plt.savefig('images/figures/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

#Display the confusion matrix image
confusion_matrix_path = Path('images/figures/confusion_matrix.png')
if confusion_matrix_path.exists():
    if sys.platform == 'darwin':  # macOS
        subprocess.run(['open', str(confusion_matrix_path)])
    elif sys.platform == 'win32':  # Windows
        subprocess.run(['start', str(confusion_matrix_path)], shell=True)
    else:  # Linux
        subprocess.run(['xdg-open', str(confusion_matrix_path)])
