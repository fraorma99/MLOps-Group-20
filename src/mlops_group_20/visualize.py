import torch
import matplotlib.pyplot as plt

history = torch.load("models/training_history.pt")
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
plt.savefig('report/figures/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
